from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os.path as osp
import sys
import pickle

from pathlib import Path
from collections import namedtuple

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import io as uio


INLIER_THRESHES = [
    0.1,
]
INLIER_RATIO_THRESHES = (np.arange(0, 21, dtype=np.float32) * 0.2 / 20).tolist()

VALID_SCENE_NAMES = []

TEST_SCENE_NAMES = [
    '7-scenes-redkitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
]

TEST_SCENE_ABBR_NAMES = [
    'Kitchen',
    'Home_1',
    'Home_2',
    'Hotel_1',
    'Hotel_2',
    'Hotel_3',
    'Study',
    'MIT_Lab',
]


Pose = namedtuple('Pose', ['indices', 'transformation'])


class RegisterResult(object):

    def __init__(self, frag1_name, frag2_name, num_inliers, inlier_ratio, gt_flag):
        self.frag1_name = frag1_name
        self.frag2_name = frag2_name
        self.num_inliers = num_inliers
        self.inlier_ratio = inlier_ratio
        self.gt_flag = gt_flag


def read_log(filepath):
    lines = uio.read_lines(filepath)
    n_poses = len(lines) // 5
    poses = list()
    for i in range(n_poses):
        items = lines[i * 5].split()  # Meta line
        id0, id1, id2 = int(items[0]), int(items[1]), int(items[2])
        mat = np.zeros((4, 4), dtype=np.float64)
        for j in range(4):
            items = lines[i * 5 + j + 1].split()
            for k in range(4):
                mat[j, k] = float(items[k])
        poses.append(Pose(indices=[id0, id1, id2], transformation=mat))
    return poses


def read_keypoints(filepath):
    return np.load(filepath)


def read_descriptors(desc_type, root_dir, scene_name, seq_name, pcd_name):
    if desc_type.startswith('MVPoolNet') or desc_type.startswith('Ours'):
        filepath = osp.join(root_dir, scene_name, seq_name, pcd_name + '.desc.npy')
        descs = np.load(filepath)
        return descs
    else:
        raise RuntimeError('[!] The descriptor type {} is not supported.'.format(desc_type))


def knn_search(points_src, points_dst, k=1):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    kdtree = o3d.geometry.KDTreeFlann(np.asarray(points_dst.T, dtype=np.float64))
    points_src = np.asarray(points_src, dtype=np.float64)
    nnindices = [
        kdtree.search_knn_vector_xd(points_src[i, :], k)[1] for i in range(len(points_src))
    ]
    if k == 1:
        return np.asarray(nnindices, dtype=np.int32)[:, 0]
    else:
        return np.asarray(nnindices, dtype=np.int32)


def register_fragment_pair(scene_name, seq_name, frag1_name, frag2_name, desc_type, poses,
                           pcloud_root, desc_root, inlier_thresh):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    print('  Start {} - {} - {} - {} - {}'.format(desc_type, scene_name, seq_name, frag1_name,
                                                  frag2_name))

    frag1_id = int(frag1_name.split('_')[-1])
    frag2_id = int(frag2_name.split('_')[-1])
    assert frag1_id < frag2_id

    overlap_pid = -1
    for pid, pose in enumerate(poses):
        if pose.indices[0] == frag1_id and pose.indices[1] == frag2_id:
            overlap_pid = pid
            break
    if overlap_pid < 0:
        num_inliers, inlier_ratio, gt_flag = 0, 0., 0
        return num_inliers, inlier_ratio, gt_flag

    frag1_pcd = o3d.io.read_point_cloud(
        osp.join(pcloud_root, scene_name, seq_name, frag1_name + '.ply'))
    frag2_pcd = o3d.io.read_point_cloud(
        osp.join(pcloud_root, scene_name, seq_name, frag2_name + '.ply'))
    frag1_kpt_indices = read_keypoints(
        osp.join(pcloud_root, scene_name, seq_name, frag1_name + '.keypts.npy'))
    frag2_kpt_indices = read_keypoints(
        osp.join(pcloud_root, scene_name, seq_name, frag2_name + '.keypts.npy'))
    frag1_kpts = np.asarray(frag1_pcd.points)[frag1_kpt_indices, :]
    frag2_kpts = np.asarray(frag2_pcd.points)[frag2_kpt_indices, :]

    frag1_descs = read_descriptors(desc_type, desc_root, scene_name, seq_name, frag1_name)
    frag2_descs = read_descriptors(desc_type, desc_root, scene_name, seq_name, frag2_name)

    assert len(frag1_kpt_indices) == len(frag1_descs)
    assert len(frag2_kpt_indices) == len(frag2_descs)

    frag21_nnindices = knn_search(frag2_descs, frag1_descs)
    assert frag21_nnindices.ndim == 1

    frag12_nnindices = knn_search(frag1_descs, frag2_descs)
    assert frag12_nnindices.ndim == 1

    frag2_match_indices = np.flatnonzero(
        np.equal(np.arange(len(frag21_nnindices)), frag12_nnindices[frag21_nnindices]))
    frag2_match_kpts = frag2_kpts[frag2_match_indices, :]
    frag1_match_kpts = frag1_kpts[frag21_nnindices[frag2_match_indices], :]

    frag2_pcd_tmp = o3d.geometry.PointCloud()
    frag2_pcd_tmp.points = o3d.utility.Vector3dVector(frag2_match_kpts)
    frag2_pcd_tmp.transform(poses[overlap_pid].transformation)

    distances = np.sqrt(
        np.sum(np.square(frag1_match_kpts - np.asarray(frag2_pcd_tmp.points)), axis=1))
    num_inliers = np.sum(distances < inlier_thresh)
    inlier_ratio = num_inliers / len(distances)
    gt_flag = 1
    return num_inliers, inlier_ratio, gt_flag


def run_scene_matching(scene_name,
                       seq_name,
                       desc_type,
                       pcloud_root,
                       desc_root,
                       out_root,
                       inlier_thresh=0.1,
                       n_threads=1):
    out_folder = osp.join(out_root, desc_type)
    uio.may_create_folder(out_folder)

    out_filename = '{}-{}-{:.2f}'.format(scene_name, seq_name, inlier_thresh)
    if Path(osp.join(out_folder, out_filename + '.pkl')).is_file():
        print('[*] {} already exists. Skip computation.'.format(out_filename))
        return osp.join(out_folder, out_filename)

    fragment_names = uio.list_files(osp.join(pcloud_root, scene_name, seq_name),
                                    '*.ply',
                                    alphanum_sort=True)
    fragment_names = [fn[:-4] for fn in fragment_names]
    n_fragments = len(fragment_names)

    register_results = [
        RegisterResult(
            frag1_name=fragment_names[i],
            frag2_name=fragment_names[j],
            num_inliers=None,
            inlier_ratio=None,
            gt_flag=None,
        ) for i in range(n_fragments) for j in range(i + 1, n_fragments)
    ]
    poses = read_log(osp.join(pcloud_root, scene_name, seq_name, 'gt.log'))

    if n_threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing

        results = Parallel(n_jobs=n_threads)(delayed(
            register_fragment_pair)(scene_name, seq_name, k.frag1_name, k.frag2_name, desc_type,
                                    poses, pcloud_root, desc_root, inlier_thresh)
                                             for k in register_results)
        for k, res in enumerate(results):
            register_results[k].num_inliers = res[0]
            register_results[k].inlier_ratio = res[1]
            register_results[k].gt_flag = res[2]
    else:
        for k in range(len(register_results)):
            num_inliers, inlier_ratio, gt_flag = register_fragment_pair(
                scene_name, seq_name, register_results[k].frag1_name,
                register_results[k].frag2_name, desc_type, poses, pcloud_root, desc_root,
                inlier_thresh)
            register_results[k].num_inliers = num_inliers
            register_results[k].inlier_ratio = inlier_ratio
            register_results[k].gt_flag = gt_flag

    with open(osp.join(out_folder, out_filename + '.pkl'), 'wb') as fh:
        to_save = {
            'register_results': register_results,
            'scene_name': scene_name,
            'seq_name': seq_name,
            'desc_type': desc_type,
            'inlier_thresh': inlier_thresh,
            'n_threads': n_threads,
        }
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(out_folder, out_filename + '.txt'), 'w') as fh:
        for k in register_results:
            fh.write('{} {} {} {:.8f} {}\n'.format(k.frag1_name, k.frag2_name, k.num_inliers,
                                                   k.inlier_ratio, k.gt_flag))

    return osp.join(out_folder, out_filename)


def compute_metrics(match_paths, desc_type, inlier_thresh, out_root, scene_abbr_fn=None):
    scenes = list()
    all_recalls = list()
    all_inliers = list()

    for match_path in match_paths:
        with open(match_path + '.pkl', 'rb') as fh:
            saved = pickle.load(fh)
            register_results = saved['register_results']
            assert saved['inlier_thresh'] == inlier_thresh
        if scene_abbr_fn is not None:
            scenes.append(scene_abbr_fn(saved['scene_name']))
        else:
            scenes.append(saved['scene_name'])

        num_inliers = list()
        inlier_ratios = list()
        gt_flags = list()
        for rr in register_results:
            num_inliers.append(rr.num_inliers)
            inlier_ratios.append(rr.inlier_ratio)
            gt_flags.append(rr.gt_flag)
        num_inliers = np.asarray(num_inliers, dtype=np.int32)
        inlier_ratios = np.asarray(inlier_ratios, dtype=np.float32)
        gt_flags = np.asarray(gt_flags, dtype=np.int32)

        recalls = list()
        inliers = list()
        for inlier_ratio_thresh in INLIER_RATIO_THRESHES:
            n_correct_matches = np.sum(inlier_ratios[gt_flags == 1] > inlier_ratio_thresh)
            recalls.append(float(n_correct_matches) / np.sum(gt_flags == 1))
            inliers.append(np.mean(num_inliers[gt_flags == 1]))
        all_recalls.append(recalls)
        all_inliers.append(inliers)

    out_path = osp.join(out_root, '{}-metrics-{:.2f}'.format(desc_type, inlier_thresh))
    with open(out_path + '.csv', 'w') as fh:
        header_str = 'SceneName'
        for inlier_ratio_thresh in INLIER_RATIO_THRESHES:
            header_str += ',Recall-{:.2f},AverageMatches-{:.2f}'.format(
                inlier_ratio_thresh, inlier_ratio_thresh)
        fh.write(header_str + '\n')

        for scene_name, recalls, inliers in zip(scenes, all_recalls, all_inliers):
            row_str = scene_name
            for recall, num_inlier in zip(recalls, inliers):
                row_str += ',{:.6f},{:.3f}'.format(recall, num_inlier)
            fh.write(row_str + '\n')

        avg_recalls = np.mean(np.asarray(all_recalls), axis=0).tolist()
        avg_inliers = np.mean(np.asarray(all_inliers), axis=0).tolist()
        avg_row_str = 'Average'
        for recall, num_inlier in zip(avg_recalls, avg_inliers):
            avg_row_str += ',{:.6f},{:.3f}'.format(recall, num_inlier)
        fh.write(avg_row_str + '\n')

    with open(out_path + '.pkl', 'wb') as fh:
        to_save = {
            'scenes': scenes,
            'recalls': all_recalls,
            'inliers': all_inliers,
            'threshes': INLIER_RATIO_THRESHES
        }
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


def plot_recall_curve(desc_types, stat_paths, out_path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    #rc('text', usetex=True)
    import matplotlib.pyplot as plt

    figure = plt.figure()
    for stat_path in stat_paths:
        with open(stat_path + '.pkl', 'rb') as fh:
            saved = pickle.load(fh)
        threshes = np.asarray(saved['threshes'])
        all_recalls = np.asarray(saved['recalls'])
        avg_recalls = np.mean(all_recalls, axis=0)
        plt.plot(threshes, avg_recalls * 100, linewidth=1)

    plt.grid(True)
    plt.xlim(0, max(threshes))
    plt.xticks(np.arange(0, 6, dtype=np.float32) * max(threshes) / 5)
    plt.ylim(0, 100)
    plt.xlabel(r'$\tau_2$')
    plt.ylabel('Recall (%)')
    plt.legend(desc_types, loc='lower left')

    figure.savefig(out_path + '.pdf', bbox_inches='tight')


def evaluate(cfg):
    assert len(cfg.desc_types) == len(cfg.desc_roots)

    if cfg.mode == 'valid':
        scene_names = VALID_SCENE_NAMES
        scene_abbr_fn = None
    elif cfg.mode == 'test':
        scene_names = TEST_SCENE_NAMES
        scene_abbr_fn = lambda sn: TEST_SCENE_ABBR_NAMES[TEST_SCENE_NAMES.index(sn)]
    else:
        raise RuntimeError('[!] Mode is not supported.')

    for inlier_thresh in INLIER_THRESHES:
        print('Start inlier_thresh {:.2f}m'.format(inlier_thresh))
        stat_paths = list()
        for desc_type, desc_root in zip(cfg.desc_types, cfg.desc_roots):
            print('  Start', desc_type)
            seq_name = 'seq-01'
            match_paths = list()
            for scene_name in scene_names:
                match_path = run_scene_matching(scene_name, seq_name, desc_type,
                                                cfg.pcloud_root, desc_root, cfg.out_root,
                                                inlier_thresh, cfg.threads)
                match_paths.append(match_path)
            stat_path = compute_metrics(match_paths, desc_type, inlier_thresh, cfg.out_root,
                                        scene_abbr_fn)
            stat_paths.append(stat_path)
        plot_recall_curve(cfg.desc_types, stat_paths,
                          osp.join(cfg.out_root, 'recall-{:.2f}'.format(inlier_thresh)))

    print('All done.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcloud_root', default='<3DMatch_Root>')
    parser.add_argument('--out_root', default='./log_3dmatch')
    parser.add_argument('--desc_roots', nargs='+')
    parser.add_argument('--desc_types', nargs='+')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--threads', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    evaluate(cfg)
