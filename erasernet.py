import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from networks import get_pad
from networks import ConvWithActivation
from networks import DeConvWithActivation
from networks1 import ConvWithActivation1
from paddle.nn import Conv2D, BatchNorm2D, ReLU6, AdaptiveAvgPool2D, Dropout
from einops import rearrange

def drop_path(x, drop_prob=0.0, training=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PSPModule(nn.Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        num_filters = num_channels // len(bin_size_list)  # C/3
        self.features = nn.LayerList()  # 一个层的空列表
        for i in range(len(bin_size_list)):
            self.features.append(
                paddle.nn.Sequential(
                    paddle.nn.AdaptiveMaxPool2D(output_size=bin_size_list[i]),
                    paddle.nn.Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=1),
                    paddle.nn.BatchNorm2D(num_features=num_filters)
                )
            )

    def forward(self, inputs, out_channels):
        # out = [inputs]  # list
        out = []
        for idx, layerlist in enumerate(self.features):
            x = layerlist(inputs)
            # 将输出上采样到与输入相同的大小
            x = paddle.nn.functional.interpolate(x=x, size=inputs.shape[2:], mode='bilinear', align_corners=True)
            out.append(x)
        # 将处理后的特征与输入连接
        out = paddle.concat(x=out, axis=1)  # NCHW
        # out = paddle.nn.Conv2D(in_channels=out.shape[1], out_channels=out_channels, kernel_size=1)(out)
        return out


class CloBlock(nn.Layer):
    def __init__(self, global_dim, local_dim, kernel_size, pool_size, head, qk_scale=None, drop_path_rate=0.0):
        super().__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.head = head

        self.dwconv1 = nn.Conv2D(
            global_dim + local_dim, global_dim + local_dim, kernel_size=3, padding=1, dilation=1, stride=1,
            groups=global_dim + local_dim)
        self.dwconv2 = nn.Conv2D(
            global_dim + local_dim, global_dim + local_dim, kernel_size=5, padding=4, dilation=2, stride=1,
            groups=global_dim + local_dim)
        self.dwconv3 = nn.Conv2D(
            global_dim + local_dim, global_dim + local_dim, kernel_size=7, padding=9, dilation=3, stride=1,
            groups=global_dim + local_dim)
        self.conv0 = nn.Conv2D((global_dim + local_dim) * 3, global_dim + local_dim, 1)

        self.norm = nn.LayerNorm(global_dim + local_dim)

        # global branch
        self.global_head = int(self.head * self.global_dim / (self.global_dim + self.local_dim))
        self.fc1 = nn.Linear(global_dim, global_dim * 3)
        self.pool1 = nn.AvgPool2D(pool_size)
        self.pool2 = nn.AvgPool2D(pool_size)
        self.qk_scale = qk_scale or global_dim ** -0.5
        self.softmax = nn.Softmax(axis=-1)

        # local branch
        self.local_head = int(self.head * self.local_dim / (self.global_dim + self.local_dim))
        self.fc2 = nn.Linear(local_dim, local_dim * 3)
        self.qconv = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
                               padding=kernel_size // 2, groups=local_dim // self.local_head)
        self.kconv = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
                               padding=kernel_size // 2, groups=local_dim // self.local_head)
        self.vconv = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, kernel_size,
                               padding=kernel_size // 2, groups=local_dim // self.local_head)
        self.fc3 = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, 1)
        self.swish = nn.Swish()
        self.fc4 = nn.Conv2D(local_dim // self.local_head, local_dim // self.local_head, 1)
        self.tanh = nn.Tanh()

        # fuse
        self.fc5 = nn.Conv2D(global_dim + local_dim, global_dim + local_dim, 1)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        identity = x
        test = x
        x1 = self.dwconv1(test)
        x2 = self.dwconv2(test)
        x3 = self.dwconv3(test)
        x = paddle.concat([x1, x2, x3], axis=1)
        x = self.conv0(x)

        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w->b (h w) c')
        x = self.norm(x)
        x_local, x_global = paddle.split(x, [self.local_dim, self.global_dim], axis=-1)

        # global branch
        global_qkv = self.fc1(x_global)
        global_qkv = rearrange(global_qkv, 'b n (m h c)->m b h n c', m=3, h=self.global_head)
        global_q, global_k, global_v = global_qkv[0], global_qkv[1], global_qkv[2]
        global_k = rearrange(global_k, 'b m (h w) c->b (m c) h w', h=H, w=W)
        global_k = self.pool1(global_k)
        global_k = rearrange(global_k, 'b (m c) h w->b m (h w) c', m=self.global_head)
        global_v = rearrange(global_v, 'b m (h w) c->b (m c) h w', h=H, w=W)
        global_v = self.pool1(global_v)
        global_v = rearrange(global_v, 'b (m c) h w->b m (h w) c', m=self.global_head)
        attn = global_q @ global_k.transpose([0, 1, 3, 2]) * self.qk_scale
        attn = self.softmax(attn)
        x_global = attn @ global_v
        x_global = rearrange(x_global, 'b m (h w) c-> b (m c) h w', h=H, w=W)

        # local branch
        local_qkv = self.fc2(x_local)
        local_qkv = rearrange(local_qkv, 'b (h w) (m n c)->m (b n) c h w', m=3, h=H, w=W, n=self.local_head)
        local_q, local_k, local_v = local_qkv[0], local_qkv[1], local_qkv[2]
        local_q = self.qconv(local_q)
        local_k = self.kconv(local_k)
        local_v = self.vconv(local_v)
        attn = local_q * local_k
        attn = self.fc4(self.swish(self.fc3(attn)))
        attn = self.tanh(attn / (self.local_dim ** -0.5))
        x_local = attn * local_v
        x_local = rearrange(x_local, '(b n) c h w->b (n c) h w', b=B)

        # Fuse
        x = paddle.concat([x_local, x_global], axis=1)
        x = self.fc5(x)
        out = identity + self.drop_path(x)
        return out


class ConvFFN(nn.Layer):
    def __init__(self, in_dim, out_dim, kernel_size, stride, exp_ratio=4, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Conv2D(in_dim, int(exp_ratio * in_dim), 1)
        self.gelu = nn.GELU()
        self.dwconv1 = nn.Conv2D(int(exp_ratio * in_dim), int(exp_ratio * in_dim), kernel_size,
                                 padding=kernel_size // 2, stride=stride, groups=int(exp_ratio * in_dim))
        self.fc2 = nn.Conv2D(int(exp_ratio * in_dim), out_dim, 1)
        self.drop_path = DropPath(drop_path_rate)

        self.downsample = stride > 1
        if self.downsample:
            self.dwconv2 = nn.Conv2D(in_dim, in_dim, kernel_size, padding=kernel_size // 2, stride=stride,
                                     groups=in_dim)
            self.norm2 = nn.BatchNorm2D(in_dim)
            self.fc3 = nn.Conv2D(in_dim, out_dim, 1)

    def forward(self, x):

        if self.downsample:
            identity = self.fc3(self.norm2(self.dwconv2(x)))
        else:
            identity = x

        x = rearrange(x, 'b c h w->b h w c')
        x = self.norm1(x)
        x = rearrange(x, 'b h w c->b c h w')

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dwconv1(x)
        x = self.fc2(x)

        out = identity + self.drop_path(x)
        return out


class Residual(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(Residual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        # if hidden_dim < oup / 6.:
        #     hidden_dim = math.ceil(oup / 6.)
        #     hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                # pw
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # nn.ReLU6(),
                # pw
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, 1, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            if self.identity_div == 1:
                out = out + x
            else:
                shape = x.shape
                id_tensor = x[:, :shape[1] // self.identity_div, :, :]
                out[:, :shape[1] // self.identity_div, :, :] = \
                    out[:, :shape[1] // self.identity_div, :, :] + id_tensor

        return out


class STRnet2_change(nn.Layer):
    def __init__(self, global_dim, local_dim, heads, depths=[1, 1], attnconv_ks=[7, 9], pool_size=[8, 4], convffn_ks=5,
                 convffn_ratio=4, drop_path_rate=0.0):
        super(STRnet2_change, self).__init__()
        self.conv1 = ConvWithActivation(3, 32, kernel_size=4, stride=2,
                                        padding=1)
        self.conva = ConvWithActivation(32, 32, kernel_size=3, stride=1,
                                        padding=1)
        self.convb = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)
        self.res1 = Residual(64, 64, 1, 1)
        self.res2 = Residual(64, 128, 2, 2)

        self.conv2 = ConvWithActivation1(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvWithActivation1(128, 128, kernel_size=3, stride=1, padding=1)

        self.deconv3 = DeConvWithActivation(128, 64, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv4 = DeConvWithActivation(64 * 2, 32, kernel_size=3,
                                            padding=1, stride=2)
        self.deconv5 = DeConvWithActivation(64, 3, kernel_size=3, padding=1,
                                            stride=2)

        self.lateral_connection3 = nn.Sequential(
            nn.Conv2D(64, 128, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(128, 64, kernel_size=1, padding=0, stride=1), )
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2D(32, 64, kernel_size=3, padding=1, stride=1),
            nn.Conv2D(64, 32, kernel_size=1, padding=0, stride=1), )
        self.conv_o1 = nn.Conv2D(64, 3, kernel_size=1)
        self.conv_o2 = nn.Conv2D(32, 3, kernel_size=1)

        self.mask_deconv_c = DeConvWithActivation(128, 64, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_c = ConvWithActivation1(64, 32, kernel_size=3,
                                               padding=1, stride=1)
        self.mask_deconv_d = DeConvWithActivation(64, 32, kernel_size=3,
                                                  padding=1, stride=2)
        self.mask_conv_d = nn.Conv2D(32, 3, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.pspmodule = PSPModule(64, [1, 2, 3, 6])
        cnum = 16
        self.astrous_net = nn.Sequential(ConvWithActivation1(4 * cnum, 4 *
                                                             cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                                         ConvWithActivation1(4 * cnum, 4 * cnum, 3, 1, dilation=4,
                                                             padding=get_pad(64, 3, 1, 4)),
                                         ConvWithActivation1(4 * cnum, 4 *
                                                             cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                                         ConvWithActivation1(4 * cnum, 4 * cnum, 3, 1, dilation=16,
                                                             padding=get_pad(64, 3, 1, 16)))
        cnum = 32
        self.coarse_conva = ConvWithActivation1(6, 32, kernel_size=4, stride=2,
                                        padding=1)
        self.coarse_convb = ConvWithActivation1(cnum, 2 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_convc = ConvWithActivation1(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)

        self.coarse_res1 = Residual(64, 128, 2, 2)
        self.coarse_res2 = Residual(128, 128, 1, 1)
        self.coarse_res3 = Residual(128, 128, 1, 1)
        self.coarse_convd = ConvWithActivation1(128, 128, kernel_size=3, stride=1, padding=1)

        self.coarse_deconva = DeConvWithActivation(4 * cnum * 2, 2 * cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_conve = ConvWithActivation1(2 * cnum, 2 * cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = DeConvWithActivation(2 * cnum * 2, cnum,
                                                   kernel_size=3, padding=1, stride=2)
        self.coarse_convf = ConvWithActivation1(cnum, cnum,
                                               kernel_size=3, stride=1, padding=1)
        self.coarse_deconvc = DeConvWithActivation(2 * cnum, 3,
                                                   kernel_size=3, padding=1, stride=2)

        self.c1 = nn.Conv2D(32, 64, kernel_size=1)
        self.c2 = nn.Conv2D(64, 128, kernel_size=1)
        self.iteration = 2

        dprs = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j < depths[i] - 1 or i == len(depths) - 1:
                    layers.append(
                        nn.Sequential(
                            CloBlock(global_dim[i], local_dim[i], attnconv_ks[i], pool_size[i], heads[i], dpr[j]),
                            ConvFFN(global_dim[i] + local_dim[i], global_dim[i] + local_dim[i], convffn_ks, 1,
                                    convffn_ratio, dpr[j])
                        )
                    )
                else:
                    layers.append(
                        nn.Sequential(
                            CloBlock(global_dim[i], local_dim[i], attnconv_ks[i], pool_size[i], heads[i], dpr[j]),
                            ConvFFN(global_dim[i] + local_dim[i], global_dim[i + 1] + local_dim[i + 1], convffn_ks, 1,
                                    convffn_ratio, dpr[j])
                        )
                    )

            self.__setattr__(f'stage{i}', nn.LayerList(layers))

    # noinspection PyShadowingNames
    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
        x = self.convb(x)
        # 这个地方一个感受野
        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        net, inp = paddle.split(x, [64, 64], axis=1)
        net = paddle.tanh(net)
        inp = F.relu(inp)
        net = self.astrous_net(net)
        for blk in self.stage0:
            inp = blk(inp)
        for blk in self.stage1:
            inp = blk(inp)
        x = paddle.concat([net, inp], axis=1)
        feature = self.conv2(x)
        # 这个地方一个感受野
        x = self.conv3(feature)
        #这个地方一个感受野
        x = self.deconv3(x)
        xo1 = x
        x = paddle.concat([self.lateral_connection3(con_x2), x], axis=1)
        x = self.deconv4(x)
        xo2 = x
        x = paddle.concat([self.lateral_connection4(con_x1), x], axis=1)
        x = self.deconv5(x)
        x_o1 = self.conv_o1(xo1)
        x_o2 = self.conv_o2(xo2)
        x_o_unet = x

        x_mask = self.pspmodule(con_x2, 64)
        mm = self.mask_deconv_c(paddle.concat([x_mask, con_x2], axis=1))
        mm = self.mask_conv_c(mm)
        mm = self.mask_deconv_d(paddle.concat([mm, con_x1], axis=1))
        mm = self.mask_conv_d(mm)
        mm = self.sig(mm)

        input = x
        for i in range(self.iteration):
            x = paddle.concat([input , x], axis=1)
            x = self.coarse_conva(x)
            x_c1 = x
            x = self.coarse_convb(x)
            x_c2 = x
            x = self.coarse_convc(x)
            x = self.coarse_res1(x)
            x_c3 = x
            x = self.coarse_res2(x)
            x = self.coarse_res3(x)
            #这个地方一个感受野
            x = self.coarse_convd(x)
            # 这个地方一个感受野
            x = self.coarse_deconva(paddle.concat([x, x_c3], axis=1))
            x = self.coarse_conve(x)
            x = self.coarse_deconvb(paddle.concat([x, x_c2], axis=1))
            x = self.coarse_convf(x)
            x = self.coarse_deconvc(paddle.concat([x, x_c1], axis=1))

        return x_o1, x_o2, x_o_unet, x, mm


def ScanEraser_xxs1():
    global_dim = [16, 32]
    local_dim = [48, 32]
    heads = [8, 16]
    model = STRnet2_change(global_dim, local_dim, heads)
    return model


if __name__ == '__main__':
    net = ScanEraser_xxs1()
    flops = paddle.flops(net, input_size=[1, 3, 512, 512], print_detail=True)