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
from evaluation.eval_geomreg_3dmatch import INLIER_THRESHES
from evaluation.eval_geomreg_3dmatch import run_scene_matching, compute_metrics, plot_recall_curve


TEST_SCENE_NAMES = ['gazebo_summer', 'gazebo_winter', 'wood_summer', 'wood_autmn']


def evaluate(cfg):
    assert len(cfg.desc_types) == len(cfg.desc_roots)

    if cfg.mode == 'test':
        scene_names = TEST_SCENE_NAMES
        scene_abbr_fn = None
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
    parser.add_argument('--pcloud_root', default='<ETH_Root>')
    parser.add_argument('--out_root', default='./log_eth')
    parser.add_argument('--desc_roots', nargs='+')
    parser.add_argument('--desc_types', nargs='+')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--threads', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    evaluate(cfg)
