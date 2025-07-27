import numpy as np
import paddle.nn as nn
from paddle.vision.ops import DeformConv2D

def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)

class ConvWithActivation1(nn.Layer):
    '''
    SN depthwise separable convolution for spectral normalization
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=nn.LeakyReLU(0.2), is_dcn=False):
        super(ConvWithActivation1, self).__init__()

        self.is_dcn = is_dcn
        if is_dcn:
            self.offsets = nn.Conv2D(in_channels, 18, kernel_size=3, stride=2,padding=1)
                    # self.mask = nn.Conv2D(in_channels, kernel_size * kernel_size, kernel_size=3, stride=stride, padding=padding)
            self.conv2d = DeformConv2D(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2d = nn.utils.spectral_norm(self.conv2d)

        else:
            # Depthwise Convolution
            self.depthwise_conv = nn.Conv2D(in_channels,in_channels,  # depthwise convolution has the same number of input and output channels
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,  # groups equals input channels for depthwise
                bias_attr=bias
            )
            self.depthwise_conv = nn.utils.spectral_norm(self.depthwise_conv)

            # Pointwise Convolution
            self.pointwise_conv = nn.Conv2D(
                in_channels,
                out_channels,  # pointwise convolution can change the number of channels
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=bias
            )
            self.pointwise_conv = nn.utils.spectral_norm(self.pointwise_conv)

        self.activation = activation
        # Weight initialization
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)

    def forward(self, input):
        if self.is_dcn:
            offsets = self.offsets(input)
            # masks = self.mask(inputs)
            x = self.conv2d(input, offsets)
        else:
            x = self.depthwise_conv(input)  # Apply depthwise convolution
            x = self.pointwise_conv(x)  # Apply pointwise convolution

        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class DeConvWithActivation1(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=1, bias=True, activation=nn.LeakyReLU(0.2)):
        super(DeConvWithActivation1, self).__init__()

        # Depthwise Transposed Convolution
        self.depthwise_conv = nn.Conv2DTranspose(
            in_channels,
            in_channels,  # depthwise transpose convolution has the same number of input and output channels
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # groups equals input channels for depthwise
            output_padding=output_padding,
            bias_attr=bias
        )
        self.depthwise_conv = nn.utils.spectral_norm(self.depthwise_conv)

        # Pointwise Transposed Convolution
        self.pointwise_conv = nn.Conv2DTranspose(
            in_channels,
            out_channels,  # pointwise transpose convolution can change the number of channels
            kernel_size=1,
            stride=1,
            padding=0,
            output_padding=0,
            bias_attr=bias
        )
        self.pointwise_conv = nn.utils.spectral_norm(self.pointwise_conv)

        self.activation = activation

    def forward(self, inputs):

        x = self.depthwise_conv(inputs)  # Apply depthwise transposed convolution
        x = self.pointwise_conv(x)  # Apply pointwise transposed convolution

        if self.activation is not None:
            return self.activation(x)
        else:
            return x