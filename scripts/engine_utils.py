import numpy as np
import os.path as osp
import random
import sys
import time
import torch
import torch.optim as optim
import warnings
import math

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.log import Logger

WEIGHT_DECAY_EXCLUDES = [
    'bias',
]
PTH_PREFIX = 'net'

def is_not_empty(x):
    return x is not None and len(x) > 0

def print_config(cfg):
    print('=========================Configurations=========================>')
    print(cfg)
    print('<================================================================')


def redirect_stdout(log_dir, prefix):
    time_stamp = time.strftime("%m_%d-%H_%M")
    sys.stdout = Logger(osp.join(log_dir, '{}-{}.log'.format(prefix, time_stamp)))


def seed_random(seed):
    seed = seed or random.randint(1, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print('[*] Using manual seed:', seed)


def get_device(gpu=None):
    if gpu is not None and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    print('[*] Using device: {}'.format(device))
    return device


def get_optimizer(name, lr, params):
    if len(params) < 1:
        return None
    if name == 'SGD':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif name == 'Adam':
        optimizer = optim.Adam(params, lr=lr)
    else:
        raise RuntimeError('[!] name is not supported.')
    return optimizer


def get_lr_scheduler(lr_step, lr_gamma, optimizer):
    if optimizer is not None and lr_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step, gamma=lr_gamma) 
    else:
        scheduler = None
    return scheduler


def get_tbwriter(log_dir):
    import warnings
    from torch.utils.tensorboard import SummaryWriter

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return SummaryWriter(log_dir, flush_secs=30)


def prepare_batch(batch, device):
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            if isinstance(batch[i], torch.Tensor):
                batch[i] = batch[i].to(device)
    return batch


def step_lr_scheduler(engine, scheduler=None):
    if scheduler is not None:
        scheduler.step()


def print_train_log(engine, timer=None, num_batches=None, cfg=None):
    state = engine.state
    iteration = state.iteration
    if iteration == 1 or iteration % cfg.log.freq == 0:
        epoch = state.epoch
        loss = state.output
        epochs = cfg.train.solver.epochs

        seconds_per_batch = timer.value()
        minutes_per_epoch = seconds_per_batch * num_batches / 60.
        total_minutes = minutes_per_epoch * epochs
        total_elapsed_minutes = seconds_per_batch * iteration / 60.
        epoch_elapsed_minutes = seconds_per_batch * (iteration % num_batches) / 60.

        msg = 'Epoch: {}/{} | Step: {}/{}'.format(epoch, epochs, iteration % num_batches,
                                                  num_batches)
        msg += ' | Iter: {}'.format(iteration)
        msg += ' | Loss: {:.5f}'.format(loss)
        msg += ' | GTime: {:.2f}/{:.2f} min'.format(total_elapsed_minutes, total_minutes)
        msg += ' | LTime: {:.2f}/{:.2f} min'.format(epoch_elapsed_minutes, minutes_per_epoch)
        print(msg)


def print_eval_log(engine, timer=None, num_batches=None):
    iteration = engine.state.iteration
    seconds_per_batch = timer.value()
    minutes_per_epoch = seconds_per_batch * num_batches / 60.
    epoch_elapsed_minutes = seconds_per_batch * (iteration % num_batches) / 60.

    msg = 'Iter: {}/{} | Time: {:.2f}/{:.2f} min'.format(iteration, num_batches,
                                                         epoch_elapsed_minutes,
                                                         minutes_per_epoch)
    print(msg)


def handle_exception(engine, e):
    if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
        engine.terminate()
        warnings.warn('[!] KeyboardInterrupt caught. Exiting gracefully.')
    else:
        raise e
