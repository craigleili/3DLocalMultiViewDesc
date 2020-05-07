from __future__ import print_function
from __future__ import division

from collections import namedtuple
from pathlib import Path
import argparse
import math
import numpy as np
import os.path as osp
import sys

ROOT_DIR = osp.abspath('../../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import io as uio

PCDMeta = namedtuple('PCDMeta', ['name', 'cloud'])


class Cloud(object):

    def __init__(self, points, indices):
        self.points = points
        self.indices = indices

    def save(self, filepath):
        np.savez(filepath, points=self.points, indices=self.indices)

    @classmethod
    def load_from(cls, filepath):
        arrays = np.load(filepath)
        return cls(arrays['points'], arrays['indices'])

    @classmethod
    def downsample_from(cls, pcd, max_points):
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points <= max_points:
            return cls(points.astype(np.float32), np.arange(n_points))
        else:
            indices = np.random.choice(n_points, max_points, replace=False)
            downsampled = points[indices, :].astype(np.float32)
            return cls(downsampled, indices)


def downsample_pcds(in_root, out_root, max_points):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    uio.may_create_folder(out_root)

    pcd_names = uio.list_files(in_root, 'cloud_bin_*.ply', alphanum_sort=True)
    pcd_stems = list()
    for pname in pcd_names:
        pstem = pname[:-4]
        pcd_path = osp.join(in_root, pname)
        pose_path = osp.join(in_root, pstem + '.pose.npy')
        pcd = o3d.io.read_point_cloud(pcd_path)
        pose = np.load(pose_path)
        pcd.transform(pose)

        down_pcd = Cloud.downsample_from(pcd, max_points)
        down_pcd.save(osp.join(out_root, pstem + '.npz'))

        pcd_stems.append(pstem)

    return pcd_stems


def compute_overlap(cfg, scene, seq, pcd_names, pid, dist_thresh=0.075):
    import pyflann

    temp_folder = osp.join(cfg.temp_root, scene, seq)
    out_folder = osp.join(cfg.out_root, scene, seq)

    n_pcds = len(pcd_names)

    pcd_src = Cloud.load_from(osp.join(temp_folder, pcd_names[pid] + '.npz'))
    n_points_src = len(pcd_src.points)
    index_src = int(pcd_names[pid][10:])
    kdtree_src = pyflann.FLANN()
    params_src = kdtree_src.build_index(pcd_src.points, algorithm='kdtree', trees=4)

    for j in range(pid + 1, n_pcds):
        pcd_dst = Cloud.load_from(osp.join(temp_folder, pcd_names[j] + '.npz'))
        n_points_dst = len(pcd_dst.points)
        index_dst = int(pcd_names[j][10:])
        assert index_src < index_dst
        if index_src + 1 == index_dst:
            continue

        knn_indices, knn_dists2 = kdtree_src.nn_index(pcd_dst.points,
                                                      num_neighbors=1,
                                                      checks=params_src['checks'])
        pair_indices = np.stack((pcd_dst.indices, pcd_src.indices[knn_indices]), axis=1)
        corr_indices = pair_indices[np.sqrt(knn_dists2) <= dist_thresh, :]

        overlap_ratio = float(len(corr_indices)) / max(n_points_src, n_points_dst)
        if overlap_ratio < 0.3:
            continue
        np.save(osp.join(out_folder, '{}-{}.npy'.format(pcd_names[j], pcd_names[pid])),
                corr_indices)


def run_seq(cfg, scene, seq):
    print("    Start {}".format(seq))

    pcd_names = downsample_pcds(osp.join(cfg.dataset_root, scene, seq),
                                osp.join(cfg.temp_root, scene, seq), cfg.max_points)
    n_pcds = len(pcd_names)

    out_folder = osp.join(cfg.out_root, scene, seq)
    if osp.exists(out_folder):
        print('    Skip...')
        return
    uio.may_create_folder(out_folder)

    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing

        Parallel(n_jobs=cfg.threads)(
            delayed(compute_overlap)(cfg, scene, seq, pcd_names, i) for i in range(n_pcds))
    else:
        for i in range(n_pcds):
            compute_overlap(cfg, scene, seq, pcd_names, i)

    print("    Finished {}".format(seq))


def run_scene(cfg, sid, scene):
    print("  Start {}th scene {} ".format(sid, scene))

    scene_folder = osp.join(cfg.dataset_root, scene)
    seqs = uio.list_folders(scene_folder, alphanum_sort=True)
    print("  {} sequences".format(len(seqs)))
    for seq in seqs:
        run_seq(cfg, scene, seq)

    print("  Finished {}th scene {} ".format(sid, scene))


def run(cfg):
    print("Start iterating dataset")

    uio.may_create_folder(cfg.out_root)

    scenes = uio.list_folders(cfg.dataset_root, alphanum_sort=False)
    print("{} scenes".format(len(scenes)))
    for sid, scene in enumerate(scenes):
        run_scene(cfg, sid, scene)

    print("Finished iterating dataset")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='<Fused_Fragments_Root>')
    parser.add_argument('--temp_root', default='./log_temp')
    parser.add_argument('--out_root', default='./log_overlaps')
    parser.add_argument('--max_points', type=int, default=100000)
    parser.add_argument('--threads', type=int, default=3)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)
