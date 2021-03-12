from torch import nn
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import AvgPool2d
from torch.nn import Softmax2d
from torch.nn import ReLU6
from torch.nn import ReLU
from torch.nn.functional import relu6
from torch.nn.functional import relu
import math


def _make_divisible(v, divisor, min_value=None):
    """
      This function is taken from the original tf repo.
      It ensures that all layers have a channel number that is divisible by 8
      It can be seen here:
      https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
      :param v:
      :param divisor:
      :param min_value:
      :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DepthSepConv(nn.Module):
    """docstring for Depthwise Separable Convolution"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 padding=1,
                 multiplier=1):
        super(DepthSepConv, self).__init__()
        in_channels = _make_divisible(in_channels * multiplier, 8)
        out_channels = _make_divisible(out_channels * multiplier, 8)
        self.depthwise_conv = Conv2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=ksize,
                                     stride=stride,
                                     padding=padding,
                                     groups=in_channels)

        self.bn1 = BatchNorm2d(in_channels)
        self.relu1 = ReLU()
        self.pointwise_conv = Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     groups=1)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu2 = ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class MobileNetV1(nn.Module):
    """
    docstring for MobileNetV1
    MobileNetV1 Body Architecture
    | Type / Stride | Filter Shape        | Input Size     | Output Size      |
    | :------------ | :------------------ | :------------- | :-------------   |
    | Conv / s2     | 3 × 3 × 3 × 32      | 224 x 224 x 3  | 112 x 112 x 32   |
    | Conv dw / s1  | 3 × 3 × 32 dw       | 112 x 112 x 32 | 112 x 112 x 32   |
    | Conv / s1     | 1 × 1 × 32 x 64     | 112 x 112 x 32 | 112 x 112 x 64   |
    | Conv dw / s2  | 3 × 3 × 64 dw       | 112 x 112 x 64 | 56 x 56 x 64     |
    | Conv / s1     | 1 × 1 × 64 × 128    | 56 x 56 x 64   | 56 x 56 x 128    |
    | Conv dw / s1  | 3 × 3 × 128 dw      | 56 x 56 x 128  | 56 x 56 x 128    |
    | Conv / s1     | 1 × 1 × 128 × 128   | 56 x 56 x 128  | 56 x 56 x 128    |
    | Conv dw / s2  | 3 × 3 × 128 dw      | 56 x 56 x 128  | 28 x 28 x 128    |
    | Conv / s1     | 1 × 1 × 128 × 256   | 28 x 28 x 128  | 28 x 28 x 256    |
    | Conv dw / s1  | 3 × 3 × 256 dw      | 28 x 28 x 256  | 28 x 28 x 256    |
    | Conv / s1     | 1 × 1 × 256 × 256   | 28 x 28 x 256  | 28 x 28 x 256    |
    | Conv dw / s2  | 3 × 3 × 256 dw      | 28 x 28 x 256  | 14 x 14 x 256    |
    | Conv / s1     | 1 × 1 × 256 × 512   | 14 x 14 x 256  | 14 x 14 x 512    |
    | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv dw / s1  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv / s1     | 1 × 1 × 512 × 512   | 14 x 14 x 512  | 14 x 14 x 512    |
    | Conv dw / s2  | 3 × 3 × 512 dw      | 14 x 14 x 512  | 7 x 7 x 512      |
    | Conv / s1     | 1 × 1 × 512 × 1024  | 7 x 7 x 512    | 7 x 7 x 1024     |
    | Conv dw / s1  | 3 × 3 × 1024 dw     | 7 x 7 x 1024   | 7 x 7 x 1024     |
    | Conv / s1     | 1 × 1 × 1024 × 1024 | 7 x 7 x 1024   | 7 x 7 x 1024     |
    | AvgPool / s1  | Pool 7 × 7          | 7 x 7 x 1024   | 1 x 1 x 1024     |
    | FC / s1       | 1024 x 1000         | 1 x 1 x 1024   | 1 x 1 x 1000     |
    | Softmax / s1  | Classifier          | 1 x 1 x 1000   | 1 x 1 x 1000     |
    """

    def __init__(self, input_size=224, n_class=1000, multiplier=1):

        super(MobileNetV1, self).__init__()
        self.name = "MobileNetV1_%d_%03d" % (input_size, int(multiplier * 100))
        assert(input_size % 32 == 0)
        self.first_in_channel = _make_divisible(32 * multiplier, 8)
        self.last_out_channel = _make_divisible(1024 * multiplier, 8)
        self.features = nn.Sequential(
            Conv2d(3, self.first_in_channel,
                   kernel_size=3, stride=2, padding=1),
            BatchNorm2d(self.first_in_channel),
            ReLU(inplace=True),
            DepthSepConv(32, 64, stride=1, multiplier=multiplier),

            DepthSepConv(64, 128, stride=2, multiplier=multiplier),
            DepthSepConv(128, 128, stride=1, multiplier=multiplier),

            DepthSepConv(128, 256, stride=2, multiplier=multiplier),
            DepthSepConv(256, 256, stride=1, multiplier=multiplier),

            DepthSepConv(256, 512, stride=2, multiplier=multiplier),
            DepthSepConv(512, 512, stride=1, multiplier=multiplier),
            DepthSepConv(512, 512, stride=1, multiplier=multiplier),
            DepthSepConv(512, 512, stride=1, multiplier=multiplier),
            DepthSepConv(512, 512, stride=1, multiplier=multiplier),
            DepthSepConv(512, 512, stride=1, multiplier=multiplier),

            DepthSepConv(512, 1024, stride=2, multiplier=multiplier),
            DepthSepConv(1024, 1024, stride=1, multiplier=multiplier))

        self._initialize_weights()


# [0:2] [2:4] [4:6] [6:12] [12 :14]     3  64  128 256 512 1024
        self.classifier = nn.Sequential(
            # 7 x 7 x 1024
            AvgPool2d(kernel_size=input_size // 32),
            # 1 x 1 x 1024
            Conv2d(self.last_out_channel, n_class, kernel_size=1),
            # 1 x 1 x n_class
            Softmax2d()
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.n_class)
        return x

    def _initialize_weights(self):
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, DepthSepConv):
                for sub in [m.depthwise_conv, m.bn1, m.pointwise_conv, m.bn2]:
                    if isinstance(sub, nn.Conv2d):
                        n = sub.kernel_size[0] * sub.kernel_size[1] * sub.out_channels
                        sub.weight.data.normal_(0, math.sqrt(2. / n))
                        if sub.bias is not None:
                            sub.bias.data.zero_()
                    elif isinstance(sub, nn.BatchNorm2d):
                        sub.weight.data.fill_(1)
                        sub.bias.data.zero_()

##################################################################################################################################################################


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, n_class=1000, multiplier=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # [3, 24, 2, 2],
        # [3, 32, 3, 2],
        # [3, 64, 4, 2],
        # [3, 96, 3, 1],
        # [3, 160, 3, 2],
        # [3, 320, 1, 1],
        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * multiplier)  # first channel is always 32!
        self.last_channel = make_divisible(
            last_channel * multiplier) if multiplier > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * multiplier) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

