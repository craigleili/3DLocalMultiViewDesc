from __future__ import division
from __future__ import print_function

from collections import defaultdict
from pathlib import Path
from scipy import spatial
import functools
import numpy as np
import os.path as osp
import pickle
import sys
import time

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import mvdesc_cfg
from data.pointclouds import PointCloudPairDataset, PointCloudDataset
from data.pointclouds import PointCloudPairSampler, list_pcd_pairs, list_pcds
from models.mvdesc import RenderModel, MV_MODELS
from scripts import engine_utils as eu
from utils import io as uio

import torch
import torchvision
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer


def prepare_config_train(cfg):
    name = uio.new_log_folder(cfg.log.root_path, cfg.log.identifier)
    name += '-{}'.format(time.strftime('%m_%d-%H_%M'))
    if cfg.render.draw_color:
        name += '-color'
    elif cfg.render.draw_depth:
        name += '-depth'
    name += '-{}'.format(cfg.model.type)
    name += '-c{}'.format(cfg.model.cnn_out_channels)
    name += '-{}'.format(cfg.model.fusion_type)
    if cfg.model.fusion_type == 'soft_pool':
        name += '-k{}'.format(cfg.view_pool.kernel)
    name += '-d{}'.format(cfg.model.desc_dim)
    name += '-{}'.format(cfg.train.dataset.name)
    if cfg.render.trainable:
        name += '-tr'
    view_num = cfg.render.view_num
    if cfg.render.augment_rotations:
        view_num *= 4
    elif cfg.render.rotation_num > 0:
        view_num *= cfg.render.rotation_num
    name += '-v{}'.format(view_num)
    if eu.is_not_empty(cfg.train.general.ckpt_path):
        name += '-ft'
    name += '-e{}'.format(cfg.train.solver.epochs)
    name += '-{}'.format(cfg.train.solver.optim)
    cfg.log.root_path = osp.join(cfg.log.root_path, name)
    uio.may_create_folder(cfg.log.root_path)


def prepare_config_eval(cfg):
    if eu.is_not_empty(cfg.eval.general.ckpt_path):
        exp_name = str(Path(cfg.eval.general.ckpt_path).parent)
    else:
        _, exp_name = uio.last_log_folder(cfg.log.root_path, cfg.log.identifier)
        ckpt_name = uio.last_checkpoint(osp.join(cfg.log.root_path, exp_name), eu.PTH_PREFIX)
        cfg.eval.general.ckpt_path = osp.join(cfg.log.root_path, exp_name, ckpt_name)

    assert eu.is_not_empty(exp_name)
    cfg.log.root_path = osp.join(cfg.log.root_path, exp_name)
    uio.may_create_folder(cfg.log.root_path)


def get_dataloader_train(cfg):
    batch_size = cfg.train.input.batch_size
    instance_num = cfg.train.input.instance_num
    name = cfg.train.dataset.name
    kpts_root = cfg.train.dataset.kpts_root
    pcloud_root = cfg.train.dataset.pcloud_root
    workers = cfg.train.dataset.workers
    radius = cfg.render.default_radius

    data = list_pcd_pairs(kpts_root)
    dataset = PointCloudPairDataset(data, pcloud_root, instance_num, radius)
    sampler = PointCloudPairSampler(data, batch_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      sampler=sampler,
                      num_workers=workers,
                      collate_fn=lambda x: x,
                      pin_memory=True,
                      drop_last=True)


def get_dataloader_eval_geomreg(cfg, mode):
    workers = cfg.eval.geomreg.workers
    radius = cfg.render.default_radius

    if mode == 'valid':
        name = cfg.eval.geomreg.valid.name
        pcloud_root = cfg.eval.geomreg.valid.pcloud_root
    elif mode == 'test':
        name = cfg.eval.geomreg.test.name
        pcloud_root = cfg.eval.geomreg.test.pcloud_root
    else:
        raise RuntimeError('[!] mode is not supported.')

    data = list_pcds(pcloud_root)
    dataset = PointCloudDataset(data, pcloud_root, radius)
    return DataLoader(dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=workers,
                      collate_fn=lambda x: x[0],
                      pin_memory=False,
                      drop_last=False)


def get_models(cfg):
    render_model = RenderModel(cfg)
    desc_model = MV_MODELS[cfg.model.type](cfg)
    return render_model, desc_model


def get_criterion(cfg):
    from utils.loss import BatchHardNegativeLoss

    return BatchHardNegativeLoss()


def step_train(engine,
               batch,
               render_model=None,
               desc_model=None,
               render_optimizer=None,
               desc_optimizer=None,
               criterion=None,
               tbwriter=None,
               device=None,
               cfg=None):
    iteration = engine.state.iteration
    grad_clip = cfg.train.solver.grad_clip
    renderer_optim_step = cfg.train.solver.renderer_optim_step
    renderer_weight = cfg.train.solver.renderer_weight
    renderer_trainable = cfg.render.trainable and iteration % renderer_optim_step == 0

    for item in batch:
        item['cloud_i'].to(device)
        item['cloud_j'].to(device)

    if renderer_trainable:
        render_optimizer.zero_grad()
    desc_optimizer.zero_grad()

    with torch.set_grad_enabled(renderer_trainable):
        renderings_i = list()
        renderings_j = list()
        for item in batch:
            cloud_i = item['cloud_i']
            cloud_j = item['cloud_j']
            renderings_i.append(
                render_model(cloud_i.points, cloud_i.radii, cloud_i.colors, cloud_i.at_centers,
                             cloud_i.at_normals))
            renderings_j.append(
                render_model(cloud_j.points, cloud_j.radii, cloud_j.colors, cloud_j.at_centers,
                             cloud_j.at_normals))
        renderings_i = torch.cat(renderings_i, dim=0)
        renderings_j = torch.cat(renderings_j, dim=0)

    with torch.set_grad_enabled(True):
        descs_i = desc_model(renderings_i)
        descs_j = desc_model(renderings_j)

        loss_batchhard = criterion(descs_i, descs_j)
        if renderer_trainable:
            loss_render = render_model.renderer.constraints()
            loss = loss_batchhard + renderer_weight * loss_render
        else:
            loss_render = None
            loss = loss_batchhard
        loss.backward()

        if renderer_trainable:
            torch.nn.utils.clip_grad_value_(
                render_model.params(True, named=False, add_prefix=False), grad_clip)
            render_optimizer.step()
        desc_optimizer.step()

    iteration = engine.state.iteration
    if iteration == 1 or iteration % cfg.log.freq == 0:
        tbwriter.add_scalar('loss', loss.item(), iteration)
        tbwriter.add_scalar('loss_batchhard', loss_batchhard.item(), iteration)
        if renderer_trainable:
            tbwriter.add_scalar('loss_render', loss_render.item(), iteration)
            tbwriter.add_scalar('lr_render', render_optimizer.param_groups[0]['lr'], iteration)
        tbwriter.add_scalar('lr', desc_optimizer.param_groups[0]['lr'], iteration)

        for name, image in zip(['renderings_i', 'renderings_j'], [renderings_i, renderings_j]):
            B, V, C, H, W = image.size()
            B = min(B, 6)
            V = cfg.render.view_num
            image_slice = image[:B, :V, :, :, :] 
            image_slice = image_slice.contiguous().view(-1, C, H, W)
            image_grid = torchvision.utils.make_grid(image_slice, nrow=V, normalize=True)
            tbwriter.add_image(name, image_grid, iteration)

    return loss.item()


def step_eval_geomreg(engine, batch, render_model=None, desc_model=None, device=None, cfg=None):
    cloud = batch['cloud']
    cloud.to(device)
    num_indices = len(cloud.at_centers)

    descs = list()
    with torch.set_grad_enabled(False):
        for i in range(num_indices):
            renderings = render_model(cloud.points, cloud.radii, cloud.colors,
                                      cloud.at_centers[[i], :], cloud.at_normals[[i], :])
            batch_desc = desc_model(renderings).cpu().numpy()
            assert batch_desc.shape[0] == 1
            descs.append(batch_desc[0, :])
    descs = np.asarray(descs, dtype=np.float32)

    scene = batch['scene']
    seq = batch['seq']
    name = batch['name']

    out_folder = osp.join(cfg.log.root_path, scene, seq)
    uio.may_create_folder(out_folder)

    np.save(osp.join(out_folder, name + '.desc.npy'), descs)
    return out_folder


def engine_train(cfg):
    prepare_config_train(cfg)

    ckpt_nets = cfg.train.general.ckpt_nets
    ckpt_path = cfg.train.general.ckpt_path
    epochs = cfg.train.solver.epochs
    gpu = cfg.general.gpu
    lr = cfg.train.solver.lr
    lr_gamma = cfg.train.solver.lr_gamma
    lr_step = cfg.train.solver.lr_step
    optim = cfg.train.solver.optim
    renderer_lr = cfg.train.solver.renderer_lr
    root_path = cfg.log.root_path
    save_freq = cfg.train.solver.save_freq
    seed = cfg.general.seed

    eu.redirect_stdout(root_path, 'train')
    eu.print_config(cfg)

    eu.seed_random(seed)

    device = eu.get_device(gpu)

    dataloader = get_dataloader_train(cfg)
    num_batches = len(dataloader)

    render_model, desc_model = get_models(cfg)
    render_model.to(device)
    render_model.train_mode()
    render_model.print_params('render_model')
    desc_model.to(device)
    desc_model.train_mode()
    desc_model.print_params('desc_model')

    crit = get_criterion(cfg)
    print('[*] Loss Function:', crit.__class__.__name__)

    render_params = render_model.params(True, named=False, add_prefix=False)
    render_optimizer = eu.get_optimizer(optim, renderer_lr, render_params)
    render_lr_scheduler = eu.get_lr_scheduler(lr_step, lr_gamma, render_optimizer)

    desc_params = desc_model.params(True, named=False, add_prefix=False)
    desc_optimizer = eu.get_optimizer(optim, lr, desc_params)
    desc_lr_scheduler = eu.get_lr_scheduler(lr_step, lr_gamma, desc_optimizer)

    if eu.is_not_empty(ckpt_path):
        render_model.load(ckpt_path, ckpt_nets)
        desc_model.load(ckpt_path, ckpt_nets)

    tbwriter = eu.get_tbwriter(root_path)

    engine = Engine(
        functools.partial(step_train,
                          render_model=render_model,
                          desc_model=desc_model,
                          render_optimizer=render_optimizer,
                          desc_optimizer=desc_optimizer,
                          criterion=crit,
                          tbwriter=tbwriter,
                          device=device,
                          cfg=cfg))
    engine.add_event_handler(Events.EPOCH_COMPLETED,
                             eu.step_lr_scheduler,
                             scheduler=render_lr_scheduler)
    engine.add_event_handler(Events.EPOCH_COMPLETED,
                             eu.step_lr_scheduler,
                             scheduler=desc_lr_scheduler)

    ckpt_handler = ModelCheckpoint(root_path,
                                   eu.PTH_PREFIX,
                                   atomic=False,
                                   save_interval=save_freq,
                                   n_saved=epochs // save_freq,
                                   require_empty=False)
    render_subnets = render_model.subnet_dict()
    desc_subnets = desc_model.subnet_dict()
    engine.add_event_handler(Events.EPOCH_COMPLETED,
                             ckpt_handler,
                             to_save={
                                 **render_subnets,
                                 **desc_subnets
                             })

    timer = Timer(average=True)
    timer.attach(engine,
                 start=Events.EPOCH_STARTED,
                 pause=Events.EPOCH_COMPLETED,
                 resume=Events.ITERATION_STARTED,
                 step=Events.ITERATION_COMPLETED)

    engine.add_event_handler(Events.ITERATION_COMPLETED,
                             eu.print_train_log,
                             timer=timer,
                             num_batches=num_batches,
                             cfg=cfg)

    engine.add_event_handler(Events.EXCEPTION_RAISED, eu.handle_exception)

    engine.run(dataloader, epochs)

    tbwriter.close()

    return root_path


def engine_eval_geomreg(cfg, mode):
    prepare_config_eval(cfg)

    ckpt_path = cfg.eval.general.ckpt_path
    gpu = cfg.general.gpu
    root_path = cfg.log.root_path
    seed = cfg.general.seed

    eu.redirect_stdout(root_path, 'eval_geomreg-{}'.format(mode))
    eu.print_config(cfg)

    eu.seed_random(seed)

    device = eu.get_device(gpu)

    dataloader = get_dataloader_eval_geomreg(cfg, mode)
    num_batches = len(dataloader)

    render_model, desc_model = get_models(cfg)
    render_model.to(device)
    render_model.eval_mode()
    render_model.print_params('render_model')
    desc_model.to(device)
    desc_model.eval_mode()
    desc_model.print_params('desc_model')

    assert eu.is_not_empty(ckpt_path)
    render_model.load(ckpt_path)
    desc_model.load(ckpt_path)

    engine = Engine(
        functools.partial(step_eval_geomreg,
                          render_model=render_model,
                          desc_model=desc_model,
                          device=device,
                          cfg=cfg))

    timer = Timer(average=True)
    timer.attach(engine,
                 start=Events.EPOCH_STARTED,
                 pause=Events.EPOCH_COMPLETED,
                 resume=Events.ITERATION_STARTED,
                 step=Events.ITERATION_COMPLETED)

    engine.add_event_handler(Events.ITERATION_COMPLETED,
                             eu.print_eval_log,
                             timer=timer,
                             num_batches=num_batches)

    engine.add_event_handler(Events.EXCEPTION_RAISED, eu.handle_exception)

    engine.run(dataloader, 1)

    return root_path


def train(cfg_path):
    cfg = mvdesc_cfg.clone()
    cfg.merge_from_file(cfg_path)
    engine_train(cfg)


def test(cfg_path):
    cfg = mvdesc_cfg.clone()
    cfg.merge_from_file(cfg_path)
    engine_eval_geomreg(cfg, 'test')


if __name__ == '__main__':
    import fire

    fire.Fire()
