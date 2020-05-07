from __future__ import division
from __future__ import print_function

from collections import defaultdict
from pathlib import Path
import cv2
import json
import numpy as np
import os
import os.path as osp
import re
import shutil


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, alphanum_sort=False):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter))]
    if alphanum_sort:
        return sorted_alphanum(file_list)
    else:
        return sorted(file_list)


def list_folders(folder_path, name_filter=None, alphanum_sort=False):
    folders = list()
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith('.'):
            folder_name = subfolder.name
            if name_filter is not None:
                if name_filter in folder_name:
                    folders.append(folder_name)
            else:
                folders.append(folder_name)
    if alphanum_sort:
        return sorted_alphanum(folders)
    else:
        return sorted(folders)


def read_lines(file_path):
    """
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines


def read_json(filepath):
    with open(filepath, 'r') as fh:
        ret = json.load(fh)
    return ret


def last_log_folder(root_folder, prefix, digits=3):
    prefix_len = len(prefix)
    tmp = list()
    for folder in list_folders(root_folder, alphanum_sort=True):
        if not folder.startswith(prefix):
            continue
        assert not is_number(folder[prefix_len + digits])
        tmp.append((int(folder[prefix_len:prefix_len + digits]), folder))
    if len(tmp) == 0:
        return 0, None
    else:
        tmp = sorted(tmp, key=lambda tup: tup[0])
        return tmp[-1][0], tmp[-1][1]


def new_log_folder(root_folder, prefix, digits=3):
    idx, _ = last_log_folder(root_folder, prefix, digits)
    tmp = prefix + '{:0' + str(digits) + 'd}'
    assert idx + 1 < 10**digits
    return tmp.format(idx + 1)


def last_checkpoint(root_folder, prefix):
    tmp = defaultdict(list)
    for file in list_files(root_folder, '{}*.pth'.format(prefix), alphanum_sort=True):
        stem = file[:-4]
        values = stem.split('_')
        tmp[values[1]].append(int(values[-1]))
    for k, v in tmp.items():
        return prefix + '_{}_' + str(sorted(v)[-1]) + '.pth'


def read_color_image(file_path):
    img = cv2.imread(file_path)
    return img[..., ::-1]


def read_gray_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img


def read_16bit_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return img


def write_color_image(file_path, image):
    cv2.imwrite(file_path, image[..., ::-1])
    return file_path


def write_gray_image(file_path, image):
    cv2.imwrite(file_path, image)
    return file_path


def write_image(file_path, image):
    if image.ndim == 2:
        return write_gray_image(file_path, image)
    elif image.ndim == 3:
        return write_color_image(file_path, image)
    else:
        raise RuntimeError('Image dimensions are not correct!')

def read_pcds(root_folder, transform):
    import open3d as o3d

    ret = dict()
    for pcd_name in list_files(root_folder, '*.pcd', alphanum_sort=True):
        pcd_path = osp.join(root_folder, pcd_name)
        pcd_stem = pcd_name[:-4]
        pcloud = o3d.io.read_point_cloud(pcd_path) 
        if transform: 
            vp_path = osp.join(root_folder, pcd_stem + '.vp.json')
            vparams = read_json(vp_path)
            modelview = np.asarray(vparams['modelview_matrix'], np.float32)
            modelview = np.reshape(modelview, (4, 4)).T
            modelview_inv = np.linalg.inv(modelview)
            pcloud.transform(modelview_inv)
        ret[pcd_stem] = np.asarray(pcloud.points)
    return ret


def write_correspondence_ply(file_path,
                             pcloudi,
                             pcloudj,
                             edges,
                             colori=(255, 255, 0),
                             colorj=(255, 0, 0),
                             edge_color=(255, 255, 255)):
    num_pointsi = len(pcloudi)
    num_pointsj = len(pcloudj)
    num_points = num_pointsi + num_pointsj
    with open(file_path, 'w') as fh:
        fh.write('ply\n')
        fh.write('format ascii 1.0\n')
        fh.write('element vertex {}\n'.format(num_points))
        fh.write('property float x\n')
        fh.write('property float y\n')
        fh.write('property float z\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('element edge {}\n'.format(len(edges)))
        fh.write('property int vertex1\n')
        fh.write('property int vertex2\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('end_header\n')

        for k in range(num_pointsi):
            fh.write('{} {} {} {} {} {}\n'.format(pcloudi[k, 0], pcloudi[k, 1], pcloudi[k, 2],
                                                  colori[0], colori[1], colori[2]))
        for k in range(num_pointsj):
            fh.write('{} {} {} {} {} {}\n'.format(pcloudj[k, 0], pcloudj[k, 1], pcloudj[k, 2],
                                                  colorj[0], colorj[1], colorj[2]))
        for k in range(len(edges)):
            fh.write('{} {} {} {} {}\n'.format(edges[k][0], edges[k][1] + num_pointsi,
                                               edge_color[0], edge_color[1], edge_color[2]))
