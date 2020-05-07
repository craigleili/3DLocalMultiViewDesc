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


def compute_radius(cfg, scene, seq, pcd_name):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    nn_radius = cfg.radius

    print('    {}'.format(pcd_name))

    pcd = o3d.io.read_point_cloud(osp.join(cfg.dataset_root, scene, seq, pcd_name))
    num_points = len(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    radii = list()
    for i in range(num_points):
        [k, nn_indices, nn_dists2] = kdtree.search_radius_vector_3d(pcd.points[i], nn_radius)
        if k < 2:
            radii.append(0)
        else:
            nn_indices = np.asarray(nn_indices)
            nn_dists2 = np.asarray(nn_dists2)
            nn_dists = np.sqrt(nn_dists2[nn_indices != i])
            radius = np.mean(nn_dists) * 0.5
            radii.append(radius)
    radii = np.asarray(radii, dtype=np.float32)
    np.save(osp.join(cfg.dataset_root, scene, seq, pcd_name[:-4] + '.radius.npy'), radii)


def run_seq(cfg, scene, seq):
    print("    Start {}".format(seq))

    pcd_names = uio.list_files(osp.join(cfg.dataset_root, scene, seq),
                               '*.ply',
                               alphanum_sort=True)
    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing

        Parallel(n_jobs=cfg.threads)(
            delayed(compute_radius)(cfg, scene, seq, pcd_name) for pcd_name in pcd_names)
    else:
        for pcd_name in pcd_names:
            compute_radius(cfg, scene, seq, pcd_name)

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

    scenes = uio.list_folders(cfg.dataset_root, alphanum_sort=False)
    print("{} scenes".format(len(scenes)))
    for sid, scene in enumerate(scenes):
        run_scene(cfg, sid, scene)

    print("Finished iterating dataset")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='<3DMatch_Fragments_Root>')
    parser.add_argument('--radius', type=float, default=0.075)   
    parser.add_argument('--threads', type=int, default=8)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)