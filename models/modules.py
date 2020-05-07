from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_module(m):
    """
    Args:
        m (nn.Module):
    """

    _linear_modules = [
        'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
        'Linear', 'Bilinear'
    ]
    _recurrent_modules = ['LSTM', 'GRU']
    _norm_modules = [
        'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d', 'InstanceNorm2d',
        'InstanceNorm3d'
    ]

    classname = m.__class__.__name__
    if classname in _norm_modules:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param.data, val=0)
            else:
                nn.init.normal_(param.data, mean=1., std=0.02)
    elif classname in _recurrent_modules:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param.data, val=0)
            else:
                nn.init.orthogonal_(param.data)
    elif classname in _linear_modules:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param.data, val=0)
            else:
                nn.init.normal_(param.data, mean=0.0, std=3.0)

def get_norm2d(norm_type='instance_norm', trainable=False):
    if norm_type == 'batch_norm':
        return functools.partial(nn.BatchNorm2d, affine=trainable)
    elif norm_type == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=trainable)
    else:
        raise NotImplementedError('[!] Normalization layer - {} is not found'.format(norm_type))


def is_batchnorm(norm_layer):
    if type(norm_layer) == functools.partial:
        return norm_layer.func == nn.BatchNorm2d
    else:
        return norm_layer.__class__.__name__ == 'BatchNorm2d'


class Conv2dNorm(nn.Module):

    def __init__(self, in_channels, out_channels, use_relu=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_relu = use_relu

        norm_layer = get_norm2d()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              bias=not is_batchnorm(norm_layer),
                              **kwargs)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.use_relu:
            return F.relu(x, inplace=True)
        else:
            return x


class ConvT2dNorm(nn.Module):

    def __init__(self, in_channels, out_channels, output_padding=0, use_relu=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.use_relu = use_relu

        norm_layer = get_norm2d()
        self.conv = nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       output_padding=output_padding,
                                       bias=not is_batchnorm(norm_layer),
                                       **kwargs)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.use_relu:
            return F.relu(x, inplace=True)
        else:
            return x


class L2NetBackbone(nn.Module):

    def __init__(self, cfg, in_channels, out_channels=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_shape = (out_channels, 8, 8)
        self.return_interims = cfg.l2net.return_interims

        self.layer1 = Conv2dNorm(in_channels,
                                 32,
                                 use_relu=True,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1) 
        self.layer2 = Conv2dNorm(32, 32, use_relu=True, kernel_size=3, stride=1,
                                 padding=1) 
        self.layer3 = Conv2dNorm(32, 64, use_relu=True, kernel_size=3, stride=2,
                                 padding=1) 
        self.layer4 = Conv2dNorm(64, 64, use_relu=True, kernel_size=3, stride=1,
                                 padding=1) 
        self.layer5 = Conv2dNorm(64, 128, use_relu=True, kernel_size=3, stride=2,
                                 padding=1) 
        self.layer6 = Conv2dNorm(128, 128, use_relu=True, kernel_size=3, stride=1,
                                 padding=1) 
        if out_channels != 128:
            self.layer7 = nn.Sequential(nn.Conv2d(128, out_channels, kernel_size=1, stride=1),
                                        nn.ReLU(inplace=True))
        else:
            self.layer7 = None

        if not cfg.l2net.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)
        e6 = self.layer6(e5)
        if self.layer7 is not None:
            e7 = self.layer7(e6)
            last = e7
        else:
            e7 = None
            last = e6
        if self.return_interims:
            if self.layer7 is not None:
                return e1, e2, e3, e4, e5, e6, e7
            else:
                return e1, e2, e3, e4, e5, e6
        else:
            return last


class L2Net(nn.Module):

    def __init__(self, cfg, in_channels):
        super().__init__()
        self.in_channels = in_channels

        backbone = L2NetBackbone(cfg, in_channels)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layer5 = backbone.layer5
        self.layer6 = backbone.layer6
        self.embed = nn.Sequential(
            nn.Dropout(0.1),
            Conv2dNorm(128, 128, use_relu=False, kernel_size=8, stride=1, padding=0))
        if not cfg.l2net.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x) 
        x = self.embed(x) 
        x = x.view(x.size(0), -1)
        x = F.normalize(x)
        return x
