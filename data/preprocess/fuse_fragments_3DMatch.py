from __future__ import print_function
from __future__ import division

from pathlib import Path
import argparse
import math
import numpy as np
import os.path as osp
import sys
import pickle

ROOT_DIR = osp.abspath('../../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import io as uio


def read_intrinsic(filepath, width, height):
    import open3d as o3d

    m = np.loadtxt(filepath, dtype=np.float32)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, m[0, 0], m[1, 1], m[0, 2],
                                                  m[1, 2])
    return intrinsic


def read_extrinsic(filepath):
    m = np.loadtxt(filepath, dtype=np.float32)
    if np.isnan(m).any():
        return None
    return m


def read_rgbd_image(cfg, color_file, depth_file, convert_rgb_to_intensity):
    import open3d as o3d

    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(
        color, depth, cfg.depth_scale, cfg.depth_trunc, convert_rgb_to_intensity)
    return rgbd_image


def process_single_fragment(cfg, color_files, depth_files, frag_id, n_frags, intrinsic_path,
                            out_folder):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    n_frames = len(color_files)
    intrinsic = read_intrinsic(intrinsic_path, cfg.width, cfg.height)

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=cfg.tsdf_cubic_size / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    sid = frag_id * cfg.frames_per_frag
    eid = min(sid + cfg.frames_per_frag, n_frames)
    pose_base2world = None
    pose_base2world_inv = None
    frag_frames = list()
    for fid in range(sid, eid):
        color_path = color_files[fid]
        depth_path = depth_files[fid]
        pose_path = color_path[:-10] + '.pose.txt'

        pose_cam2world = read_extrinsic(pose_path)
        if pose_cam2world is None:
            continue
        if fid == sid:
            pose_base2world = pose_cam2world
            pose_base2world_inv = np.linalg.inv(pose_base2world)
        if pose_base2world_inv is None:
            break
        pose_cam2world = np.matmul(pose_base2world_inv, pose_cam2world)

        rgbd = read_rgbd_image(cfg, color_path, depth_path, False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose_cam2world))

        frag_frames.append(color_path[:-10])

    if pose_base2world_inv is None:
        return

    pcloud = volume.extract_point_cloud()
    o3d.geometry.estimate_normals(pcloud)
    o3d.io.write_point_cloud(osp.join(out_folder, 'cloud_bin_{}.ply'.format(frag_id)), pcloud)

    np.save(osp.join(out_folder, 'cloud_bin_{}.pose.npy'.format(frag_id)), pose_base2world)

    with open(osp.join(out_folder, 'cloud_bin_{}.frames.pkl'.format(frag_id)), 'wb') as fh:
        to_save = {'frames': frag_frames}
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)


def run_seq(cfg, scene, seq):
    print("    Start {}".format(seq))

    seq_folder = osp.join(cfg.dataset_root, scene, seq)
    color_names = uio.list_files(seq_folder, '*.color.png', alphanum_sort=True)
    color_paths = [osp.join(seq_folder, cf) for cf in color_names]
    depth_paths = [osp.join(seq_folder, cf[:-10] + '.depth.png') for cf in color_names]

    n_frames = len(color_paths)
    n_frags = int(math.ceil(float(n_frames) / cfg.frames_per_frag))

    out_folder = osp.join(cfg.out_root, scene, seq)
    uio.may_create_folder(out_folder)

    intrinsic_path = osp.join(cfg.dataset_root, scene, 'camera-intrinsics.txt')

    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing

        Parallel(n_jobs=cfg.threads)(delayed(process_single_fragment)(
            cfg, color_paths, depth_paths, frag_id, n_frags, intrinsic_path, out_folder)
                                     for frag_id in range(n_frags))

    else:
        for frag_id in range(n_frags):
            process_single_fragment(cfg, color_paths, depth_paths, frag_id, n_frags,
                                    intrinsic_path, out_folder)

    print("    Finished {}".format(seq))


def run_scene(cfg, scene):
    print("  Start scene {} ".format(scene))

    scene_folder = osp.join(cfg.dataset_root, scene)
    seqs = uio.list_folders(scene_folder, alphanum_sort=True)
    print("  {} sequences".format(len(seqs)))
    for seq in seqs:
        run_seq(cfg, scene, seq)

    print("  Finished scene {} ".format(scene))


def run(cfg):
    print("Start iterating dataset")

    uio.may_create_folder(cfg.out_root)

    scenes = uio.list_folders(cfg.dataset_root, alphanum_sort=False)
    print("{} scenes".format(len(scenes)))
    for scene in scenes:
        run_scene(cfg, scene)

    print("Finished iterating dataset")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='<3DMatch_RGBD_Root>')
    parser.add_argument('--out_root', default='./log_fragments')
    parser.add_argument('--depth_scale', type=float, default=1000.0)
    parser.add_argument('--depth_trunc', type=float, default=6.0)
    parser.add_argument('--frames_per_frag', type=int, default=50)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--tsdf_cubic_size', type=float, default=3.0)
    parser.add_argument('--width', type=int, default=640)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)


# Scene list:
#
# 7-scenes-chess
# 7-scenes-fire
# 7-scenes-heads
# 7-scenes-office
# 7-scenes-pumpkin
# 7-scenes-stairs
# bundlefusion-apt0
# bundlefusion-apt1
# bundlefusion-apt2
# bundlefusion-copyroom
# bundlefusion-office0
# bundlefusion-office1
# bundlefusion-office2
# bundlefusion-office3
# rgbd-scenes-v2-scene_01
# rgbd-scenes-v2-scene_02
# rgbd-scenes-v2-scene_03
# rgbd-scenes-v2-scene_04
# rgbd-scenes-v2-scene_05
# rgbd-scenes-v2-scene_06
# rgbd-scenes-v2-scene_07
# rgbd-scenes-v2-scene_08
# rgbd-scenes-v2-scene_09
# rgbd-scenes-v2-scene_10
# rgbd-scenes-v2-scene_11
# rgbd-scenes-v2-scene_12
# rgbd-scenes-v2-scene_13
# rgbd-scenes-v2-scene_14
# sun3d-harvard_c5-hv_c5_1
# sun3d-harvard_c6-hv_c6_1
# sun3d-harvard_c8-hv_c8_3
# sun3d-home_bksh-home_bksh_oct_30_2012_scan2_erika
# sun3d-hotel_nips2012-nips_4
# sun3d-hotel_sf-scan1
# sun3d-mit_32_d507-d507_2
# sun3d-mit_46_ted_lab1-ted_lab_2
