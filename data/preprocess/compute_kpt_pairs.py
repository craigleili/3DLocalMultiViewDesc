from __future__ import print_function
from __future__ import division

from collections import namedtuple, defaultdict
from pathlib import Path
import argparse
import math
import numpy as np
import os.path as osp
import sys
import random
import pickle

ROOT_DIR = osp.abspath('../../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import io as uio


def downsample_and_compute_fpfh(cfg, scene, seq, pcd_name):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    print('    {}'.format(pcd_name))

    temp_folder = osp.join(cfg.temp_root, scene, seq)

    pcd = o3d.io.read_point_cloud(osp.join(cfg.dataset_root, scene, seq, pcd_name))
    pcd.normalize_normals()
    pcd_down = o3d.geometry.voxel_down_sample(pcd, cfg.voxel_size)

    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamRadius(cfg.fpfh_radius))

    pose = np.load(osp.join(cfg.dataset_root, scene, seq, pcd_name[:-4] + '.pose.npy'))
    pcd_down.transform(pose)

    o3d.io.write_point_cloud(osp.join(temp_folder, pcd_name), pcd_down)
    np.save(osp.join(temp_folder, pcd_name[:-4] + '.fpfh.npy'), np.asarray(pcd_fpfh.data).T)


def match_fpfh(cfg, scene, seq, pcd_stem_i, pcd_stem_j):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    print('    {} - {}'.format(pcd_stem_i, pcd_stem_j))

    temp_folder = osp.join(cfg.temp_root, scene, seq)

    pcd_down_i = o3d.io.read_point_cloud(osp.join(temp_folder, pcd_stem_i + '.ply'))
    pcd_down_j = o3d.io.read_point_cloud(osp.join(temp_folder, pcd_stem_j + '.ply'))
    points_i = np.asarray(pcd_down_i.points)
    points_j = np.asarray(pcd_down_j.points)

    fpfh_i = np.load(osp.join(temp_folder, pcd_stem_i + '.fpfh.npy'))
    fpfh_j = np.load(osp.join(temp_folder, pcd_stem_j + '.fpfh.npy'))

    indices_i = np.arange(len(points_i))[np.any(fpfh_i != 0, axis=1)]
    indices_j = np.arange(len(points_j))[np.any(fpfh_j != 0, axis=1)]

    fpfh_i = fpfh_i[indices_i, :]
    fpfh_j = fpfh_j[indices_j, :]
    points_i = points_i[indices_i, :]
    points_j = points_j[indices_j, :]

    kdtree_j = o3d.geometry.KDTreeFlann(fpfh_j.T)
    nnindices = [
        kdtree_j.search_knn_vector_xd(fpfh_i[k, :], 1)[1][0] for k in range(len(fpfh_i))
    ]
    points_j = points_j[nnindices, :]

    distances = np.sqrt(np.sum(np.square(points_i - points_j), axis=1))
    match_flags = distances <= cfg.dist_thresh

    if np.sum(match_flags) < 128: return

    points_i = points_i[match_flags, :]
    points_j = points_j[match_flags, :]

    pair_indices = list()
    for pcd_stem, query_points in zip([pcd_stem_i, pcd_stem_j], [points_i, points_j]):
        pcd = o3d.io.read_point_cloud(osp.join(cfg.dataset_root, scene, seq, pcd_stem + '.ply'))
        pose = np.load(osp.join(cfg.dataset_root, scene, seq, pcd_stem + '.pose.npy'))
        pcd.transform(pose)

        kdtree = o3d.geometry.KDTreeFlann(np.asarray(pcd.points).T)
        nnindices = [
            kdtree.search_knn_vector_3d(query_points[k, :], 1)[1][0]
            for k in range(len(query_points))
        ]
        pair_indices.append(np.asarray(nnindices))

    pair_indices = np.stack(pair_indices, axis=1)
    out_npy_path = osp.join(cfg.out_root, scene, seq,
                            '{}-{}.npy'.format(pcd_stem_i, pcd_stem_j))
    np.save(out_npy_path, pair_indices)


def collate_kpts(cfg, scene, seq):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    kpt_pair_folder = osp.join(cfg.out_root, scene, seq)

    pcd_kpt_indices = defaultdict(list)
    for npy_file in uio.list_files(kpt_pair_folder, '*.npy', True):
        pcd_stem_i, pcd_stem_j = npy_file[:-4].split('-')
        kpt_pairs = np.load(osp.join(kpt_pair_folder, npy_file))
        pcd_kpt_indices[pcd_stem_i].extend(kpt_pairs[:, 0].tolist())
        pcd_kpt_indices[pcd_stem_j].extend(kpt_pairs[:, 1].tolist())
    if len(pcd_kpt_indices) < 1:
        return

    scene_points = list()
    scene_normals = list()
    labels = list()
    for pcd_stem, kpt_indices in pcd_kpt_indices.items():
        pcd = o3d.io.read_point_cloud(osp.join(cfg.dataset_root, scene, seq, pcd_stem + '.ply'))
        pose = np.load(osp.join(cfg.dataset_root, scene, seq, pcd_stem + '.pose.npy'))
        pcd.transform(pose) 
        pcd.normalize_normals()

        uni_kpt_indices = list(set(kpt_indices))
        scene_points.append(np.asarray(pcd.points)[uni_kpt_indices, :])
        scene_normals.append(np.asarray(pcd.normals)[uni_kpt_indices, :])
        labels.extend(list(zip([pcd_stem] * len(uni_kpt_indices), uni_kpt_indices)))
    scene_points = np.concatenate(scene_points, axis=0)
    scene_normals = np.concatenate(scene_normals, axis=0)

    print('    {} scene points/normals'.format(len(scene_points)))

    kdtree = o3d.geometry.KDTreeFlann(scene_points.T)
    num_points = len(scene_points)
    flags = [False] * num_points
    identities = list()
    for i in range(num_points):
        if flags[i]: continue

        [_, nn_indices,
         nn_dists2] = kdtree.search_radius_vector_3d(scene_points[i, :], cfg.dist_thresh)
        nn_indices = [j for j in nn_indices if not flags[j]] 

        nn_normal = [scene_normals[j] for j in nn_indices]
        if len(nn_normal) < 2: continue
        nn_normal = np.mean(np.asarray(nn_normal), axis=0)
        nn_normal /= np.linalg.norm(nn_normal)

        nn_pcd_indices = defaultdict(list)
        for j in nn_indices:
            if np.arccos(np.clip(np.dot(scene_normals[j], nn_normal), -1,
                                 1)) > cfg.angle_thresh:
                continue
            nn_pcd_indices[labels[j][0]].append(labels[j][1])
        if len(nn_pcd_indices) < 2: continue

        identities.append({k: random.choice(v) for k, v in nn_pcd_indices.items()})

        for j in nn_indices:
            flags[j] = True
        flags[i] = True

    with open(osp.join(cfg.out_root, scene, '{}.kpts.pkl'.format(seq)), 'wb') as fh:
        to_save = {'identities': identities}
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print('    {} identities'.format(len(identities)))


def run_seq(cfg, scene, seq):
    print('    Start {}'.format(seq))

    out_folder = osp.join(cfg.out_root, scene, seq)
    if osp.exists(out_folder):
        print('    Skip...')
        return
    uio.make_clean_folder(out_folder)

    temp_folder = osp.join(cfg.temp_root, scene, seq)
    uio.make_clean_folder(temp_folder)

    print('    Start downsampling and computing FPFH')
    pcd_names = uio.list_files(osp.join(cfg.dataset_root, scene, seq),
                               'cloud_bin_*.ply',
                               alphanum_sort=True)
    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing

        Parallel(n_jobs=cfg.threads)(
            delayed(downsample_and_compute_fpfh)(cfg, scene, seq, pcd_name)
            for pcd_name in pcd_names)
    else:
        for pcd_name in pcd_names:
            downsample_and_compute_fpfh(cfg, scene, seq, pcd_name)

    print('    Start matching FPFH')
    overlaps = uio.list_files(osp.join(cfg.overlap_root, scene, seq),
                              'cloud_bin_*.npy',
                              alphanum_sort=True)
    overlap_pcds = [npy_file[:-4].split('-') for npy_file in overlaps]
    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing

        Parallel(n_jobs=cfg.threads)(
            delayed(match_fpfh)(cfg, scene, seq, pcd_pair[0], pcd_pair[1])
            for pcd_pair in overlap_pcds)
    else:
        for pcd_pair in overlap_pcds:
            match_fpfh(cfg, scene, seq, pcd_pair[0], pcd_pair[1])

    print('    Start collating kpts')
    collate_kpts(cfg, scene, seq)

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
    parser.add_argument('--dataset_root', default='<Fused_Fragments_Root>')
    parser.add_argument('--overlap_root', default='./log_overlaps')
    parser.add_argument('--out_root', default='./log_kpts')
    parser.add_argument('--temp_root', default='./log_temp')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--fpfh_radius', type=float, default=0.15)
    parser.add_argument('--dist_thresh', type=float, default=0.03)
    parser.add_argument('--angle_thresh', type=float, default=math.pi / 12.0)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)
