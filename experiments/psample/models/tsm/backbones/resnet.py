"""
TODO: abandon TSM's 2D Conv, using 3D Conv with kernel size (1, k, k)
for all Conv modules!
"""
from collections import OrderedDict

import torch
import torch.nn as nn
# from torchvision.models.resnet import BasicBlock, Bottleneck
from torchstream.ops import TemporalShift

from .utils import load_state_dict_from_url


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def tshift_conv3x3(in_planes, out_planes, seg_num, stride=1, groups=1,
                  dilation=1, fold_div=8, shift_steps=1):
    """3x3 convolution with padding and temporal shift"""
    shift = TemporalShift(seg_num=seg_num,
                          fold_div=fold_div,
                          shift_steps=shift_steps)
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)
    return nn.Sequential(
        OrderedDict([
            ("shift", shift),
            ("conv", conv)
        ]))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, bias=False)


def tshift_conv1x1(in_planes, out_planes, seg_num, stride=1,
                  fold_div=8, shift_steps=1):
    """1x1 convolution with temporal shift"""
    shift = TemporalShift(seg_num=seg_num,
                          fold_div=fold_div,
                          shift_steps=shift_steps)
    conv = nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, bias=False)
    return nn.Sequential(
        OrderedDict([
            ("shift", shift),
            ("conv", conv)
        ]))


class BasicTemporalShiftBlock(nn.Module):
    """
    Args:
        seg_num: temporal extent
        fold_div
        shift_steps
    """
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, seg_num,
                 stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 fold_div=8, shift_steps=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1"
                             " and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported"
                                      "in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = tshift_conv3x3(inplanes=inplanes, planes=planes,
                                    seg_num=seg_num, stride=stride,
                                    fold_div=fold_div,
                                    shift_steps=shift_steps)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tshift_conv3x3(inplanes=inplanes, planes=planes,
                                    seg_num=seg_num, stride=stride,
                                    fold_div=fold_div,
                                    shift_steps=shift_steps)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TemporalShiftBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,  seg_num,
                 stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 fold_div=8, shift_steps=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = tshift_conv1x1(in_planes=inplanes, out_planes=width,
                                    seg_num=seg_num,
                                    fold_div=fold_div,
                                    shift_steps=shift_steps)
        self.bn1 = norm_layer(width)
        self.conv2 = tshift_conv3x3(in_planes=width, out_planes=width,
                                    seg_num=seg_num, stride=stride,
                                    groups=groups, dilation=dilation,
                                    fold_div=fold_div,
                                    shift_steps=shift_steps)
        self.bn2 = norm_layer(width)
        self.conv3 = tshift_conv1x1(in_planes=width,
                                    out_planes=planes * self.expansion,
                                    seg_num=seg_num,
                                    fold_div=fold_div,
                                    shift_steps=shift_steps)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TSMResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (Bottleneck, TemporalShiftBottleneck)):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, (BasicBlock, BasicTemporalShiftBlock)):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = TSMResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""TSM ResNet-18
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400
        progress (bool): If True, displays downloading progress bar to stderr
    """
    return _resnet(arch='resnet18',
                   block=BasicTemporalShiftBlock,
                   layers=[2, 2, 2, 2],
                   pretrained=pretrained,
                   progress=progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""TSM ResNet-34
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400
        progress (bool): If True, displays downloading progress bar to stderr
    """
    return _resnet(arch='resnet34',
                   block=BasicTemporalShiftBlock,
                   layers=[3, 4, 6, 3],
                   pretrained=pretrained,
                   progress=progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400
        progress (bool): If True, displays downloading progress bar to stderr
    """
    return _resnet(arch='resnet50',
                   block=TemporalShiftBottleneck,
                   layers=[3, 4, 6, 3],
                   pretrained=pretrained,
                   progress=progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""TSM ResNet-101
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400
        progress (bool): If True, displays downloading progress bar to stderr
    """
    return _resnet(arch='resnet101',
                   block=TemporalShiftBottleneck,
                   layers=[3, 4, 23, 3],
                   pretrained=pretrained,
                   progress=progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""TSM ResNet-152
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400
        progress (bool): If True, displays downloading progress bar to stderr
    """
    return _resnet(arch='resnet152',
                   block=TemporalShiftBottleneck,
                   layers=[3, 8, 36, 3],
                   pretrained=pretrained,
                   progress=progress,
                   **kwargs)
