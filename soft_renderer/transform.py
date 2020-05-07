from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from soft_renderer.cuda.jit import soft_rasterize_cuda as src


def rotate_about(a, b, theta):
    a = a.view(1, 1, 1, -1) 
    b = b.view(b.size(0), 1, b.size(1), b.size(3))  
    theta = theta.view(1, -1, 1, 1) 

    ab_dot = torch.sum(a * b, dim=3, keepdim=True) 
    bb_dot = torch.sum(b * b, dim=3, keepdim=True) 
    par = (ab_dot / bb_dot) * b 
    perp = a - par 
    w = torch.cross(b, perp, dim=3) 

    perp_norm = torch.norm(perp, p=2, dim=3, keepdim=True) 
    res = par + perp * torch.cos(theta) + perp_norm * F.normalize(w, dim=3) * torch.sin(
        theta) 
    return res.view(res.size(0), -1, 1, res.size(3))


def init_eye_xyz_coords(num_views,
                        rho_min=0.3,
                        rho_max=1.0,
                        phi_min=0.,
                        phi_max=math.pi / 2.,
                        theta_min=0.,
                        theta_max=2. * math.pi):
    shape = (num_views,)
    rho = np.random.uniform(rho_min, rho_max, shape)
    phi = np.random.uniform(phi_min, phi_max, shape)
    theta = np.random.uniform(theta_min, theta_max, shape)

    local_x = rho * np.sin(phi) * np.cos(theta) 
    local_y = rho * np.sin(phi) * np.sin(theta) 
    local_z = rho * np.cos(phi) 
    eye = np.stack((local_x, local_y, local_z), axis=1) 
    return eye


def spherical_to_xyz_coords(rho, phi, theta):
    x = rho * torch.sin(phi) * torch.cos(theta) 
    y = rho * torch.sin(phi) * torch.sin(theta) 
    z = rho * torch.cos(phi) 
    xyz = torch.stack((x, y, z), dim=1) 
    return xyz


def look_at(centers, normals, up, eye_lcs, thetas=None):
    V = centers.size(0)
    N = eye_lcs.size(0)
    up = torch.unsqueeze(up, dim=0) 

    axis_z = F.normalize(normals, dim=1) 
    axis_y = F.normalize(up, dim=1).repeat(V, 1) 
    axis_x = F.normalize(torch.cross(axis_y, axis_z, dim=1), dim=1) 
    axis_y = F.normalize(torch.cross(axis_z, axis_x, dim=1), dim=1) 
    axis = torch.cat((torch.unsqueeze(axis_x, dim=1), torch.unsqueeze(
        axis_y, dim=1), torch.unsqueeze(axis_z, dim=1)),
                     dim=1) 
    centers = centers.view(-1, 1, 1, 3) 
    eye = torch.matmul(eye_lcs.view(1, -1, 1, 3), axis.view(-1, 1, 3,
                                                            3)) + centers 

    front_dir = F.normalize(eye - centers, dim=3) 
    up_dir = F.normalize(up, dim=1)
    if thetas is None:
        up_dir = up.view(1, 1, 1, -1).repeat(V, N, 1, 1) 
        right_dir = F.normalize(torch.cross(up_dir, front_dir, dim=3), dim=3) 
        up_dir = F.normalize(torch.cross(front_dir, right_dir, dim=3), dim=3) 

        mat = torch.cat((right_dir, up_dir, front_dir), dim=2) 
        vec = -1. * torch.matmul(mat, torch.transpose(eye, 2, 3)) 
        mat = torch.cat((mat, vec), dim=3) 

        vec = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=centers.dtype,
                           device=centers.device) 
        mat = torch.cat((mat, vec.view(1, 1, 1, 4).repeat(V, N, 1, 1)), dim=2) 
    else:
        R = thetas.size(0)
        up_dir = rotate_about(up_dir, front_dir, thetas) 
        front_dir = front_dir.repeat(1, R, 1, 1) 
        right_dir = F.normalize(torch.cross(up_dir, front_dir, dim=3), dim=3) 
        up_dir = F.normalize(torch.cross(front_dir, right_dir, dim=3), dim=3) 

        mat = torch.cat((right_dir, up_dir, front_dir), dim=2) 
        eye = eye.repeat(1, R, 1, 1) 
        vec = -1. * torch.matmul(mat, torch.transpose(eye, 2, 3)) 
        mat = torch.cat((mat, vec), dim=3) 
        vec = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=centers.dtype,
                           device=centers.device) 
        mat = torch.cat((mat, vec.view(1, 1, 1, 4).repeat(V, N * R, 1, 1)),
                        dim=2) 
    return mat


def look(eye, direction, up):
    V, N = eye.size(0), eye.size(1)
    front_dir = F.normalize(-direction, dim=2) 
    up_dir = F.normalize(up.view(1, 1, 3), dim=2) 
    up_dir = up_dir.repeat(V, N, 1) 
    right_dir = F.normalize(torch.cross(up_dir, front_dir, dim=2), dim=2) 
    up_dir = F.normalize(torch.cross(front_dir, right_dir, dim=2), dim=2) 

    mat = torch.cat((torch.unsqueeze(right_dir, dim=2), torch.unsqueeze(
        up_dir, dim=2), torch.unsqueeze(front_dir, dim=2)),
                    dim=2) 
    vec = -1. * torch.matmul(mat, torch.unsqueeze(eye, dim=3)) 
    mat = torch.cat((mat, vec), dim=3) 

    vec = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=mat.dtype, device=mat.device) 
    mat = torch.cat((mat, vec.view(1, 1, 1, 4).repeat(V, N, 1, 1)), dim=2) 
    return mat


def projection_matrix(field_of_view, aspect, znear, zfar):
    mat = np.zeros((4, 4), dtype=np.float32)
    fov_rad = field_of_view / 180.0 * math.pi
    tan_half_fov = math.tan(fov_rad / 2.0)
    mat[0, 0] = 1.0 / aspect / tan_half_fov
    mat[1, 1] = 1.0 / tan_half_fov
    mat[2, 2] = -(zfar + znear) / (zfar - znear)
    mat[3, 2] = -1.0
    mat[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    return torch.from_numpy(mat)


class SoftRasterizeFunc(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            mvps, 
            vertices, 
            radii, 
            colors, 
            sigma, 
            gamma,
            dist_ratio,
            znear,
            zfar,
            tan_half_fov,
            image_size,
            compute_weight,
            draw_color,
            draw_depth):

        assert draw_color != draw_depth

        V, N = mvps.size(0), mvps.size(1)
        num_images = V * N
        device = mvps.device
        dtype = mvps.dtype

        if colors is None:
            colors = torch.tensor([], dtype=dtype, device=device)
        if compute_weight:
            weights = torch.ones(
                num_images, 1, image_size, image_size, dtype=dtype, device=device) * 1e-8
        else:
            weights = torch.tensor([], dtype=dtype, device=device)
        if draw_color:
            color_map = torch.zeros(num_images,
                                    3,
                                    image_size,
                                    image_size,
                                    dtype=dtype,
                                    device=device)
        else:
            color_map = torch.tensor([], dtype=dtype, device=device)
        if draw_depth:
            depth_map = torch.zeros(num_images,
                                    1,
                                    image_size,
                                    image_size,
                                    dtype=dtype,
                                    device=device)
        else:
            depth_map = torch.tensor([], dtype=dtype, device=device)

        pseudo_depth_map = torch.ones(num_images,
                                      1,
                                      image_size,
                                      image_size,
                                      dtype=dtype,
                                      device=device)
        locks = torch.zeros_like(pseudo_depth_map, dtype=torch.int32)

        src.soft_rasterize_forward(mvps.view(-1, 4, 4), vertices, radii, colors, locks, sigma,
                                   gamma, dist_ratio, znear, zfar, tan_half_fov, image_size,
                                   compute_weight, draw_color, draw_depth, weights, color_map,
                                   depth_map, pseudo_depth_map)

        ctx.save_for_backward(mvps, vertices, radii, colors, weights)
        ctx.sigma = sigma
        ctx.gamma = gamma
        ctx.dist_ratio = dist_ratio
        ctx.znear = znear
        ctx.zfar = zfar
        ctx.tan_half_fov = tan_half_fov
        ctx.image_size = image_size
        ctx.compute_weight = compute_weight
        ctx.draw_color = draw_color
        ctx.draw_depth = draw_depth

        if draw_color:
            return color_map.view(V, N, -1, image_size, image_size)
        if draw_depth:
            return depth_map.view(V, N, -1, image_size, image_size)

    @staticmethod
    def backward(ctx, grad_map):
        mvps, vertices, radii, colors, weights = ctx.saved_tensors
        V, N = mvps.size(0), mvps.size(1)
        num_images = V * N
        device = mvps.device
        dtype = mvps.dtype

        if ctx.draw_color:
            grad_color_map = grad_map.view(num_images, -1, ctx.image_size, ctx.image_size)
        else:
            grad_color_map = torch.tensor([], dtype=dtype, device=device)
        if ctx.draw_depth:
            grad_depth_map = grad_map.view(num_images, -1, ctx.image_size, ctx.image_size)
        else:
            grad_depth_map = torch.tensor([], dtype=dtype, device=device)

        grad_mvps = torch.zeros_like(mvps) 

        src.soft_rasterize_backward(mvps.view(-1, 4, 4), vertices, radii, colors, weights,
                                    grad_color_map, grad_depth_map, ctx.sigma, ctx.gamma,
                                    ctx.dist_ratio, ctx.znear, ctx.zfar, ctx.tan_half_fov,
                                    ctx.image_size, ctx.draw_color, ctx.draw_depth, grad_mvps)

        return grad_mvps, None, None, None, None, None, None, None, None, None, None, None, None, None


class MultiViewRenderer(nn.Module):

    def __init__(self,
                 num_views,
                 num_rotations=0,
                 field_of_view=60.,
                 aspect=1.,
                 znear=0.1,
                 zfar=6.,
                 image_size=64,
                 sigma=1. / 64.,
                 gamma=5.,
                 dist_ratio=5.,
                 dist_factor=1.0,
                 radius_ratio=0.25,
                 draw_color=False,
                 draw_depth=True,
                 trainable=True):
        super().__init__()
        self.num_views = num_views
        self.num_rotations = num_rotations
        self.field_of_view = field_of_view
        self.aspect = aspect
        self.znear = znear
        self.zfar = zfar
        self.image_size = image_size
        self.sigma = sigma
        self.gamma = gamma
        self.dist_ratio = dist_ratio
        self.dist_factor = dist_factor
        self.radius_ratio = radius_ratio
        self.draw_color = draw_color
        self.draw_depth = draw_depth
        self.trainable = trainable
        self.tan_half_fov = math.tan(field_of_view / 180.0 * math.pi / 2.0)

        self.rho_min = 0.3
        self.rho_max = 1.0
        self.phi_min = 0
        self.phi_max = math.pi / 2.
        self.theta_min = 0
        self.theta_max = 2. * math.pi
        self.rot_min = 0
        self.rot_max = 2. * math.pi

        proj_mat = projection_matrix(field_of_view, aspect, znear, zfar)
        self.register_buffer('proj_mat', proj_mat)

        rho = np.random.uniform(self.rho_min, self.rho_max, (num_views,))
        phi = np.random.uniform(self.phi_min, self.phi_max, (num_views,))
        theta = np.random.uniform(self.theta_min, self.theta_max, (num_views,))

        rho = torch.tensor(rho, dtype=torch.float32)
        phi = torch.tensor(phi, dtype=torch.float32)
        theta = torch.tensor(theta, dtype=torch.float32)
        up = torch.tensor([0., -1., 0.], dtype=torch.float32)
        if self.num_rotations > 0:
            rot = torch.tensor(np.random.uniform(self.rot_min, self.rot_max, (num_rotations,)),
                               dtype=torch.float32)
        if self.trainable:
            self.rho = nn.Parameter(rho)
            self.phi = nn.Parameter(phi)
            self.theta = nn.Parameter(theta)
            self.up = nn.Parameter(up)
            if self.num_rotations > 0:
                self.rot = nn.Parameter(rot)
        else:
            self.register_buffer('rho', rho)
            self.register_buffer('phi', phi)
            self.register_buffer('theta', theta)
            self.register_buffer('up', up)
            if self.num_rotations > 0:
                self.register_buffer('rot', rot)

    def constraints(self):
        if not self.trainable:
            raise RuntimeError('Renderer is not trainable.')

        params = [self.rho, self.phi, self.theta]
        min_vals = [self.rho_min, self.phi_min, self.theta_min]
        max_vals = [self.rho_max, self.phi_max, self.theta_max]
        if self.num_rotations > 0:
            params.append(self.rot)
            min_vals.append(self.rot_min)
            max_vals.append(self.rot_max)

        res = None
        for i in range(len(params)):
            mid = torch.ones_like(params[i]) * ((min_vals[i] + max_vals[i]) / 2.)
            rng = torch.ones_like(params[i]) * ((max_vals[i] - min_vals[i]) / 2.)
            diff = torch.mean(F.relu(torch.abs(params[i] - mid) - rng))
            if res is None:
                res = diff
            else:
                res += diff
        return res

    def forward(self, vertices, radii, colors, at_centers, at_normals):
        compute_weight = self.trainable and self.training

        rho = torch.clamp(self.rho, self.rho_min, self.rho_max)
        phi = torch.clamp(self.phi, self.phi_min, self.phi_max)
        theta = torch.clamp(self.theta, self.theta_min, self.theta_max)
        if self.num_rotations > 0:
            rot = torch.clamp(self.rot, self.rot_min, self.rot_max)
        else:
            rot = None
        eye_lcs = spherical_to_xyz_coords(rho * self.dist_factor, phi, theta)
        mv_mat = look_at(at_centers, at_normals, self.up, eye_lcs, rot)
        mvp_mat = torch.matmul(self.proj_mat.view(1, 1, 4, 4), mv_mat)

        images = SoftRasterizeFunc.apply(mvp_mat, vertices, radii * self.radius_ratio, colors,
                                         self.sigma, self.gamma, self.dist_ratio, self.znear,
                                         self.zfar, self.tan_half_fov, self.image_size,
                                         compute_weight, self.draw_color, self.draw_depth)
        return images, mv_mat
