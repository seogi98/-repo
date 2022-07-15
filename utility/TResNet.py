"""
TResNet: High Performance GPU-Dedicated Architecture
https://arxiv.org/pdf/2003.13630.pdf

Original model: https://github.com/mrT23/TResNet

"""
import copy
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SpaceToDepthModule, AntiAliasDownsampleLayer, InplaceAbn, ClassifierHead, SEModule

def IABN2Float(module: nn.Module) -> nn.Module:
    "If `module` is IABN don't use half precision."
    if isinstance(module, InplaceAbn):
        module.float()
    for child in module.children(): IABN2Float(child)
    return module


def conv2d_iabn(ni, nf, stride, kernel_size=3, groups=1, act_layer="leaky_relu", act_param=1e-2):
    return nn.Sequential(
        nn.Conv2d(
            ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False),
        # InplaceAbn(nf, act_layer=act_layer, act_param=act_param)
        nn.BatchNorm2d(nf),
        nn.LeakyReLU()

    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, aa_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv1 = conv2d_iabn(inplanes, planes, stride=2, act_param=1e-3)
            else:
                self.conv1 = nn.Sequential(
                    conv2d_iabn(inplanes, planes, stride=1, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_iabn(planes, planes, stride=1, act_layer="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduction_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduction_channels=reduction_chs) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True,
                 act_layer="leaky_relu", aa_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_iabn(
            inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer, act_param=1e-3)
        if stride == 1:
            self.conv2 = conv2d_iabn(
                planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3)
        else:
            if aa_layer is None:
                self.conv2 = conv2d_iabn(
                    planes, planes, kernel_size=3, stride=2, act_layer=act_layer, act_param=1e-3)
            else:
                self.conv2 = nn.Sequential(
                    conv2d_iabn(planes, planes, kernel_size=3, stride=1, act_layer=act_layer, act_param=1e-3),
                    aa_layer(channels=planes, filt_size=3, stride=2))

        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduction_channels=reduction_chs) if use_se else None

        self.conv3 = conv2d_iabn(
            planes, planes * self.expansion, kernel_size=1, stride=1, act_layer="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, no_aa_jit=False,
                 global_pool='fast', drop_rate=0.):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        aa_layer = partial(AntiAliasDownsampleLayer, no_jit=no_aa_jit)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_iabn(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(
            BasicBlock, self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer)  # 56x56
        layer2 = self._make_layer(
            BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer)  # 28x28
        layer3 = self._make_layer(
            Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer)  # 14x14
        layer4 = self._make_layer(
            Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        self.feature_info = [
            dict(num_chs=self.planes, reduction=2, module=''),  # Not with S2D?
            dict(num_chs=self.planes, reduction=4, module='body.layer1'),
            dict(num_chs=self.planes * 2, reduction=8, module='body.layer2'),
            dict(num_chs=self.planes * 4 * Bottleneck.expansion, reduction=16, module='body.layer3'),
            dict(num_chs=self.planes * 8 * Bottleneck.expansion, reduction=32, module='body.layer4'),
        ]

        # head
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InplaceAbn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_iabn(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=1, act_layer="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, use_se=use_se, aa_layer=aa_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, use_se=use_se, aa_layer=aa_layer))
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='fast'):
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def TResnetM(model_params):
    """ Constructs a medium TResnet model.
    """
    in_chans = 3
    num_classes = model_params.n_classes
    model = TResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans)
    return model


def TResnetL(model_params):
    """ Constructs a large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['n_classes']
    model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.2)
    return model


def TResnetXL(model_params):
    """ Constructs an extra-large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['n_classes']
    model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.3)

    return model
