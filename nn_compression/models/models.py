import torch
import torch.nn as nn


def model_resnet18_seg(class_count, **kwargs):
    return ResNetSeg(BasicDsBlock1, [2, 2, 2, 2],
                     BasicUsBlock1, [1, 1, 1, 1],
                     class_count,
                     **kwargs)

def model_cnn1_seg(class_count, **kwargs):
    return Cnn1_seg(3, class_count)


class Cnn1_seg(nn.Module):
    def __init__(self, input_channels, class_count):
        super(Cnn1_seg, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, class_count, kernel_size=3, padding=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()
        x = self.layers.forward(x)
        x = nn.functional.interpolate(x, input_size[2:4], mode='bilinear')
        return x

def _conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicDsBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicDsBlock1, self).__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def _conv(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(
        nn.BatchNorm2d(in_planes),
        nn.ReLU(True),
        nn.Conv2d(in_planes, out_planes, kernel_size, **kwargs)
    )

class BasicDsBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicDsBlock2, self).__init__()
        self.conv1 = _conv(in_planes, out_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = _conv(out_planes, out_planes, kernel_size=3, padding=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class BasicUsBlock1(nn.Module):
    contraction = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicUsBlock1, self).__init__()

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_planes, out_planes // self.contraction,
                kernel_size=1, stride=stride, output_padding=1)
        )

    def forward(self, x):
        return self.upsample(x)

class BasicUsBlock2(nn.Module):
    contraction = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicUsBlock2, self).__init__()

        self.stride = stride

    def forward(self, x):
        x = nn.functional.interpolate(x,
            scale_factor=self.stride, mode='bilinear')
        return x


class BasicUsBlock3(nn.Module):
    contraction = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicUsBlock3, self).__init__()

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_planes, out_planes // self.contraction,
                kernel_size=stride, stride=stride)
        )

    def forward(self, x):
        return self.upsample(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetSeg(nn.Module):

    def __init__(self, ds_block, ds_layers, us_block, us_layers, num_classes):
        super(ResNetSeg, self).__init__()

        ds_channels = [64, 128, 256, 512]
        us_channels = [64, 128, 256, 512]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_ds_stage(ds_block, ds_channels[0], ds_layers[0])
        self.layer2 = self._make_ds_stage(ds_block, ds_channels[1], ds_layers[1], stride=2)
        self.layer3 = self._make_ds_stage(ds_block, ds_channels[2], ds_layers[2], stride=2)
        self.layer4 = self._make_ds_stage(ds_block, ds_channels[3], ds_layers[3], stride=2)

        self.us_stage4 = self._make_us_stage(us_block, us_channels[3], us_layers[3], stride=2)
        self.us_stage3 = self._make_us_stage(us_block, us_channels[2], us_layers[2], stride=2)
        self.us_stage2 = self._make_us_stage(us_block, us_channels[1], us_layers[1], stride=2)
        self.us_stage1 = self._make_us_stage(us_block, us_channels[0], us_layers[0], stride=2)
        self.inplanes = us_channels[0] // us_block.contraction // 2

        self.classifier = _conv(self.inplanes, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_ds_stage(self, block, out_planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != out_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * block.expansion),
            )
            # downsample =_conv(self.inplanes, out_planes * block.expansion,
            #     kernel_size=1, stride=stride, bias=False)

        layers = [ block(self.inplanes, out_planes, stride, downsample) ]

        self.inplanes = out_planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_planes))

        return nn.Sequential(*layers)


    def _make_us_stage(self, block, in_planes, blocks, stride=1):
        planes = in_planes // stride

        layers = [ block(in_planes, planes, stride) ]

        in_planes = planes // block.contraction
        for _ in range(1, blocks):
            layers.append(block(in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        stage1 = self.layer1(x)
        stage2 = self.layer2(stage1)
        stage3 = self.layer3(stage2)
        stage4 = self.layer4(stage3)

        x = self.us_stage4(stage4)[:, :, 0:stage3.size()[2], 0:stage3.size(3)]
        x += stage3
        x = self.us_stage3(x)[:, :, 0:stage2.size()[2], 0:stage2.size(3)]
        x += stage2
        x = self.us_stage2(x)[:, :, 0:stage1.size()[2], 0:stage1.size(3)]
        x += stage1
        x = self.us_stage1(x)

        x = self.classifier(x)

        x = nn.functional.interpolate(x, input_size[2:4], mode='bilinear')

        return x