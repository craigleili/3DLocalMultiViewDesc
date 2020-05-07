from __future__ import division
from __future__ import print_function

import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.base_model import BaseModel
from models.modules import L2NetBackbone
from models.modules import Conv2dNorm, ConvT2dNorm, init_module
from soft_renderer.transform import MultiViewRenderer


class BaseViewFusionModule(nn.Module):

    def __init__(self, cfg, in_shape):
        super().__init__()

        self.cfg = cfg
        assert len(in_shape) == 4
        self.in_shape = in_shape
        self.out_channels = None

        self._init()

    def _init(self):
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError


class ViewPoolFusion(BaseViewFusionModule):
    def _init(self):
        self.out_channels = self.in_shape[1] 

    def forward(self, x):
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)  
        else:
            x, _ = torch.max(x, dim=1) 
            return x


class SoftViewPoolFusion(BaseViewFusionModule):
    def _init(self):
        in_channels = self.in_shape[1]
        self.out_channels = in_channels

        kernel = self.cfg.view_pool.kernel
        bias = self.cfg.view_pool.bias
        if kernel == 1:
            self.attn = nn.Sequential(
                nn.Conv2d(in_channels,
                          in_channels // 2,
                          kernel_size=kernel,
                          stride=1,
                          bias=bias), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2,
                          in_channels,
                          kernel_size=kernel,
                          stride=1,
                          bias=bias))
        elif kernel == 3:
            self.attn = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels // 2,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    bias=bias,
                ), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 2,
                                   in_channels,
                                   kernel_size=kernel,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=bias))
        else:
            raise RuntimeError('[!] cfg.view_pool.kernel={} is not supported.'.format(kernel))

    def forward(self, x):
        B, V, C, H, W = x.size()
        a = self.attn(x.view(B * V, C, H, W)) 
        a = F.softmax(a.view(B, V, C, H, W), dim=1)
        ax = torch.sum(a * x, dim=1)
        return ax


class BaseMVDescModel(BaseModel):
    BACKBONES = {'l2net': L2NetBackbone}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cnn_output = None
        self.fusion_output = None

        self._init()

    def _init(self):
        raise NotImplementedError

    def get_cnn_backbone(self):
        draw_color = self.cfg.render.draw_color
        draw_depth = self.cfg.render.draw_depth
        cnn = self.cfg.model.cnn
        cnn_out_channels = self.cfg.model.cnn_out_channels

        image_channels = -1
        if draw_color:
            image_channels = 3
        elif draw_depth:
            image_channels = 1
        else:
            raise RuntimeError('[!] Cannot decide image channels.')

        backbone = self.BACKBONES[cnn]
        return backbone(self.cfg, image_channels, cnn_out_channels)

    @staticmethod
    def _call_backbone(cnn, x):
        B, V, C, H, W = x.size()
        x = x.view(B * V, C, H, W)
        x = cnn(x)
        C, H, W = x.size(1), x.size(2), x.size(3)
        x = x.view(B, V, C, H, W)
        return x


class MVPoolNet(BaseMVDescModel):
    POOLS = {'max_pool': ViewPoolFusion, 'soft_pool': SoftViewPoolFusion}

    def _init(self):
        fusion_type = self.cfg.model.fusion_type
        desc_dim = self.cfg.model.desc_dim
        view_num = self.cfg.render.view_num
        augment_rotations = self.cfg.render.augment_rotations
        rotation_num = self.cfg.render.rotation_num
        if augment_rotations:
            view_num *= 4
        elif rotation_num > 0:
            view_num *= rotation_num

        # Subnets
        self.cnn = self.get_cnn_backbone()
        C, H, W = self.cnn.out_shape

        pool_fn = self.POOLS[fusion_type]
        self.pool = pool_fn(self.cfg, (view_num, C, H, W))
        C = self.pool.out_channels
        print('[*] Using', self.pool.__class__.__name__)

        self.embed = nn.Sequential(
            nn.Conv2d(C, desc_dim, kernel_size=(H, W), stride=1, padding=0))

        self.register_nets([self.cnn, self.pool, self.embed], ['cnn', 'pool', 'embed'],
                           [True] * 3)

    def __call__(self, x):
        x = self._call_backbone(self.cnn, x)
        self.cnn_output = x

        x = self.pool(x)
        self.fusion_output = x

        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x


MV_MODELS = {
    'MVPoolNet': MVPoolNet,
}


class RenderModel(BaseModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.renderer = MultiViewRenderer(cfg.render.view_num,
                                          cfg.render.rotation_num,
                                          znear=cfg.render.znear,
                                          zfar=cfg.render.zfar,
                                          image_size=cfg.render.image_size,
                                          sigma=cfg.render.sigma,
                                          gamma=cfg.render.gamma,
                                          dist_ratio=cfg.render.dist_ratio,
                                          dist_factor=cfg.render.dist_factor,
                                          radius_ratio=cfg.render.radius_ratio,
                                          draw_color=cfg.render.draw_color,
                                          draw_depth=cfg.render.draw_depth,
                                          trainable=cfg.render.trainable)

        self.register_nets([self.renderer], ['renderer'], [cfg.render.trainable])

    def __call__(self, vertices, radii, colors, at_centers, at_normals):
        images, _ = self.renderer(vertices, radii, colors, at_centers, at_normals)
        if self.cfg.render.augment_rotations:
            images = self._augment_rotation(images)
        return images

    @staticmethod
    def _augment_rotation(x):
        res = [x]
        for i in range(3):
            res.append(torch.rot90(x, k=i + 1, dims=(3, 4)))
        res = torch.cat(res, dim=1)
        return res
