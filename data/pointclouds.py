from __future__ import division
from __future__ import print_function

from collections import defaultdict, namedtuple
from pathlib import Path
import math
import numpy as np
import open3d as o3d 
import os.path as osp
import random
import sys
import pickle

import torch
from torch.utils.data import Dataset, Sampler

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import io as uio
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

CAMERA_UP = np.asarray([0., -1., 0.], dtype=np.float32)
OverlapMeta = namedtuple('OverlapMeta',
                         ['scene', 'seq', 'cloud_name_i', 'cloud_name_j', 'full_path'])
PCloudMeta = namedtuple('PCloudMeta', ['scene', 'seq', 'name', 'full_path'])


def is_numpy(x):
    if x is None:
        return False
    return type(x).__module__ == np.__name__


def random_index(n, excludes=[]):
    while True:
        i = random.randint(0, n - 1)
        if i not in excludes:
            return i


def list_pcd_pairs(root_dir, excluded_scenes=None):
    res = list()
    for scene in uio.list_folders(root_dir, alphanum_sort=False):
        if excluded_scenes is not None and scene in excluded_scenes:
            continue
        for seq in uio.list_folders(osp.join(root_dir, scene), alphanum_sort=True):
            seq_folder = osp.join(root_dir, scene, seq)
            for npy_file in uio.list_files(seq_folder, 'cloud_bin_*.npy', alphanum_sort=True):
                cloud_name_i, cloud_name_j = npy_file[:-4].split('-')
                res.append(
                    OverlapMeta(scene=scene,
                                seq=seq,
                                cloud_name_i=cloud_name_i,
                                cloud_name_j=cloud_name_j,
                                full_path=osp.join(seq_folder, npy_file)))
    return res


def list_pcds(root_dir, excluded_scenes=None):
    res = list()
    for scene in uio.list_folders(root_dir, alphanum_sort=False):
        if excluded_scenes is not None and scene in excluded_scenes:
            continue
        for seq in uio.list_folders(osp.join(root_dir, scene), alphanum_sort=True):
            seq_folder = osp.join(root_dir, scene, seq)
            pcloud_names = uio.list_files(seq_folder, '*.ply', alphanum_sort=True)
            metas = [
                PCloudMeta(
                    scene=scene,
                    seq=seq,
                    name=pn[:-4],
                    full_path=osp.join(seq_folder, pn),
                ) for pn in pcloud_names
            ]
            res.extend(metas)
    return res


class PointCloud(object):

    def __init__(self, points, radii, colors, at_centers, at_normals):
        assert points is not None
        assert radii is not None
        assert at_centers is not None
        assert at_normals is not None

        if is_numpy(points):
            self.points = torch.from_numpy(points)
        else:
            self.points = points

        if is_numpy(radii):
            self.radii = torch.from_numpy(radii)
        else:
            self.radii = radii

        if is_numpy(colors):
            self.colors = torch.from_numpy(colors)
        else:
            self.colors = colors

        if is_numpy(at_centers):
            self.at_centers = torch.from_numpy(at_centers)
        else:
            self.at_centers = at_centers

        if is_numpy(at_normals):
            self.at_normals = torch.from_numpy(at_normals)
        else:
            self.at_normals = at_normals

    def to(self, device):
        self.points = self.points.to(device)
        self.radii = self.radii.to(device)
        if self.colors is not None:
            self.colors = self.colors.to(device)
        self.at_centers = self.at_centers.to(device)
        self.at_normals = self.at_normals.to(device)

    @classmethod
    def from_o3d(cls, pcd_o3d, radii, at_indices, at_normals=None):
        points = np.asarray(pcd_o3d.points, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        if len(pcd_o3d.colors) == len(pcd_o3d.points):
            colors = np.asarray(pcd_o3d.colors, dtype=np.float32)
        else:
            colors = None
        at_centers = points[at_indices, :]
        if at_normals is None:
            if len(pcd_o3d.normals) != len(pcd_o3d.points):
                raise RuntimeError('[!] The point cloud needs normals.')
            at_normals = np.asarray(pcd_o3d.normals, dtype=np.float32)[at_indices, :]
        return cls(points, radii, colors, at_centers, at_normals)


class PointCloudPairSampler(Sampler):

    def __init__(self, data_source, batch_size=1):
        self.data_source = data_source
        self.batch_size = batch_size

        self.indices = self._generate_iter_indices()
        self.regen_flag = False

    def __iter__(self):
        if self.regen_flag:
            self.indices = self._generate_iter_indices()
        else:
            self.regen_flag = True
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def _generate_iter_indices(self):
        indices_dict = dict()
        for i, meta in enumerate(self.data_source):
            scene, seq = meta.scene, meta.seq
            if scene not in indices_dict:
                indices_dict[scene] = dict()
            if seq not in indices_dict[scene]:
                indices_dict[scene][seq] = list()
            indices_dict[scene][seq].append(i)

        grouped_indices = list()
        for scene_name, scene_dict in indices_dict.items():
            for seq_name, seq_list in scene_dict.items():
                meta_indices = seq_list.copy()
                random.shuffle(meta_indices)
                n_sublists = math.floor(float(len(meta_indices)) / self.batch_size)
                for i in range(n_sublists):
                    grouped_indices.append(meta_indices[i * self.batch_size:(i + 1) *
                                                        self.batch_size])
        random.shuffle(grouped_indices)
        iter_indices = list()
        for item in grouped_indices:
            iter_indices.extend(item)
        return iter_indices


class PointCloudPairDataset(Dataset):
    def __init__(self, data_source, pcd_root, num_point_pairs, radius=None):
        self.data_source = data_source
        self.pcd_root = pcd_root
        self.num_point_pairs = num_point_pairs
        self.radius = radius

    def __getitem__(self, index):
        meta = self.data_source[index]

        path_i = osp.join(self.pcd_root, meta.scene, meta.seq, meta.cloud_name_i + '.ply')
        path_j = osp.join(self.pcd_root, meta.scene, meta.seq, meta.cloud_name_j + '.ply')

        pcd_o3d_i = o3d.io.read_point_cloud(path_i)
        pcd_o3d_j = o3d.io.read_point_cloud(path_j)
        if Path(path_i[:-4] + '.radius.npy').is_file() and Path(path_j[:-4] + '.radius.npy').is_file():
            radii_i = np.load(path_i[:-4] + '.radius.npy')
            radii_j = np.load(path_j[:-4] + '.radius.npy')
        else:
            assert self.radius is not None
            radii_i = np.ones((len(pcd_o3d_i.points),), dtype=np.float32) * self.radius
            radii_j = np.ones((len(pcd_o3d_j.points),), dtype=np.float32) * self.radius

        point_pairs = np.load(meta.full_path)
        samples = random.sample(range(len(point_pairs)), self.num_point_pairs)
        indices = point_pairs[samples, :]

        pcd_i = PointCloud.from_o3d(pcd_o3d_i, radii_i, indices[:, 0], None)
        pcd_j = PointCloud.from_o3d(pcd_o3d_j, radii_j, indices[:, 1], None)

        return {
            'cloud_i': pcd_i,
            'cloud_j': pcd_j,
            'name_i': '{}/{}/{}'.format(meta.scene, meta.seq, meta.cloud_name_i),
            'name_j': '{}/{}/{}'.format(meta.scene, meta.seq, meta.cloud_name_j),
        }

    def __len__(self):
        return len(self.data_source)


class PointCloudDataset(Dataset):

    def __init__(self, data_source, pcd_root, radius=None):
        self.data_source = data_source
        self.pcd_root = pcd_root
        self.radius = radius

    def __getitem__(self, index):
        meta = self.data_source[index]
        pcd_o3d = o3d.io.read_point_cloud(meta.full_path)
        if Path(meta.full_path[:-4] + '.radius.npy').is_file():
            radii = np.load(meta.full_path[:-4] + '.radius.npy')
        else:
            assert self.radius is not None
            radii = np.ones((len(pcd_o3d.points),), dtype=np.float32) * self.radius
        kpt_indices = np.load(meta.full_path[:-4] + '.keypts.npy')
        pcd = PointCloud.from_o3d(pcd_o3d, radii, kpt_indices)

        return {
            'cloud': pcd,
            'scene': meta.scene,
            'seq': meta.seq,
            'name': meta.name,
        }

    def __len__(self):
        return len(self.data_source)
