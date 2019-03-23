import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class EnhDropout(nn.Module):
    def __init__(self, prob=0.5, dropout=nn.Dropout2d):
        super(EnhDropout, self).__init__()
        self.dropout = dropout(prob)

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
            # x /= self.dropout.p
        return x

def _conv(inplanes, planes, *args, inplace=True, dropout=None, **kwargs):
    do = lambda: [ EnhDropout(prob=dropout, dropout=nn.Dropout2d) ] if dropout else []

    return [
        *do(),
        nn.Conv2d(inplanes, planes, *args, **kwargs),
        nn.BatchNorm2d(planes),
        nn.ReLU(inplace)
    ]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, dropout=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate
        self.dropout = EnhDropout(prob=dropout, dropout=nn.Dropout2d) if dropout else None

    def forward(self, x):
        residual = x

        if self.dropout:
            x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.dropout:
            x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x

class Bottleneck_sep(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, dropout=None, **kwargs):
        super(Bottleneck_sep, self).__init__()
        self.main_block = nn.Sequential(
            *_conv(inplanes, planes, kernel_size=1, bias=False, inplace=True, dropout=dropout),
            *_conv(planes, planes, kernel_size=(3,1), stride=(stride,1),
                dilation=(rate,1), padding=(rate,0), bias=False, inplace=True, dropout=dropout),
            *_conv(planes, planes, kernel_size=(1,3), stride=(1,stride),
                dilation=(1,rate), padding=(0,rate), bias=False, inplace=True, dropout=dropout),
            *_conv(planes, planes * 4, kernel_size=1, bias=False, dropout=dropout)[0:-1]
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.main_block(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False, channels_sep=1, **kwargs):
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        channels = [v // channels_sep for v in  [64, 128, 256, 512]]
        self.inplanes = channels[0]

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], rate=rates[0], **kwargs)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], rate=rates[1], **kwargs)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], rate=rates[2], **kwargs)
        self.layer4 = self._make_MG_unit(block, channels[3], blocks=blocks, stride=strides[3], rate=rates[3], **kwargs)
        
        self.out_channels = self.inplanes

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, rate=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def ResNet101(nInputChannels=3, os=16, pretrained=False, **kwargs):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained, **kwargs)
    return model

def ResNet101_Sep(nInputChannels=3, os=16, pretrained=False, **kwargs):
    model = ResNet(nInputChannels, Bottleneck_sep, [3, 4, 23, 3], os, pretrained=pretrained, **kwargs)
    return model


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, dropout=None, **kwargs):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

        self.dropout = EnhDropout(prob=dropout, dropout=nn.Dropout2d) if dropout else None

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP_module_sep(nn.Module):
    def __init__(self, inplanes, planes, rate, dropout=None, **kwargs):
        super(ASPP_module_sep, self).__init__()

        if rate == 1:
            self.layers = nn.Sequential(
                *_conv(inplanes, planes, kernel_size=1,
                    stride=1, padding=0, dilation=rate, 
                    bias=False, inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                *_conv(inplanes, planes, kernel_size=(3,1),
                    stride=1, padding=(rate,0), dilation=(rate,1), 
                    bias=False, inplace=True, dropout=dropout),
                *_conv(planes, planes, kernel_size=(1,3),
                    stride=1, padding=(0,rate), dilation=(1,rate), 
                    bias=False, inplace=True, dropout=dropout),
            )

        self._init_weight()

    def forward(self, x):
        return self.layers(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16,
            pretrained=False, _print=True,
            backbone=ResNet101, aspp=ASPP_module,
            **kwargs):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        dropout = kwargs.get('dropout', None)
        self.dropout = EnhDropout(prob=dropout, dropout=nn.Dropout2d) if dropout else None

        # Atrous Conv
        self.resnet_features = backbone(nInputChannels, os, pretrained=pretrained, **kwargs)
        backbone_output_channels = self.resnet_features.out_channels

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        channels_sep = kwargs.get('head_channels_sep', 1)
        self.aspp1 = aspp(backbone_output_channels, 256 // channels_sep, rate=rates[0], **kwargs)
        self.aspp2 = aspp(backbone_output_channels, 256 // channels_sep, rate=rates[1], **kwargs)
        self.aspp3 = aspp(backbone_output_channels, 256 // channels_sep, rate=rates[2], **kwargs)
        self.aspp4 = aspp(backbone_output_channels, 256 // channels_sep, rate=rates[3], **kwargs)

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(backbone_output_channels, 256 // channels_sep, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256 // channels_sep),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280 // channels_sep, 256 // channels_sep, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256 // channels_sep)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256 // kwargs.get('channels_sep', 1), 48 // channels_sep, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48 // channels_sep)

        dropout = lambda: [ self.dropout ] if self.dropout else []
        self.last_conv = nn.Sequential(*dropout(),
                                       nn.Conv2d(304 // channels_sep, 256 // channels_sep, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256 // channels_sep),
                                       nn.ReLU(),
                                       *dropout(),
                                       nn.Conv2d(256 // channels_sep, 256 // channels_sep, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256 // channels_sep),
                                       nn.ReLU(),
                                       nn.Conv2d(256 // channels_sep, n_classes, kernel_size=1, stride=1))

    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        if self.dropout:
            x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        if self.dropout:
            low_level_features = self.dropout(low_level_features)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

def model_deeplabv3(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=True, _print=True)

def model_deeplabv3_do(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=True, _print=True, dropout=0.5)

def model_deeplabv3_do_soft(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=True, _print=True, dropout=0.25)

def model_deeplabv3_do_hard(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=True, _print=True, dropout=0.75)

def model_deeplabv3_reduced2(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, channels_sep=2)

def model_deeplabv3_do_reduced2(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, dropout=0.5, channels_sep=2)

def model_deeplabv3_do_reduced2_2(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, dropout=0.5, channels_sep=2, head_channels_sep=2)

def model_deeplabv3_sep_backbone(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, backbone=ResNet101_Sep)

def model_deeplabv3_sep_aspp(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, backbone=ResNet101, aspp=ASPP_module_sep)

def model_deeplabv3_sep_backbone_and_aspp(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, backbone=ResNet101_Sep, aspp=ASPP_module_sep)

def model_deeplabv3_sep_backbone_and_aspp_do(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, backbone=ResNet101_Sep, aspp=ASPP_module_sep, dropout=0.5)

def model_deeplabv3_sep_backbone_and_aspp_do_reduced2(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, backbone=ResNet101_Sep, aspp=ASPP_module_sep, dropout=0.5, channels_sep=2)

def model_deeplabv3_sep_backbone_and_aspp_do_reduced2_2(class_count, **kwargs):
    return DeepLabv3_plus(nInputChannels=3, n_classes=class_count, os=16,
        pretrained=False, _print=True, backbone=ResNet101_Sep, aspp=ASPP_module_sep, dropout=0.5, channels_sep=2, head_channels_sep=2)
