from __future__ import division
from yacs.config import CfgNode as CN

M = CN()

M.general = CN()
M.general.gpu = 0
M.general.note = ''
M.general.seed = 9

M.log = CN()
M.log.freq = 100  
M.log.identifier = ''
M.log.root_path = ''

M.render = CN()
M.render.augment_rotations = True
M.render.view_num = 1
M.render.rotation_num = 0
M.render.znear = 0.1
M.render.zfar = 6.
M.render.image_size = 64
M.render.sigma = 1. / 64.
M.render.gamma = 5.
M.render.dist_ratio = 3.
M.render.radius_ratio = 0.5
M.render.draw_color = False
M.render.draw_depth = True
M.render.trainable = True
M.render.default_radius = 0.025
M.render.dist_factor = 1.0

M.model = CN()
M.model.cnn = 'l2net'
M.model.cnn_out_channels = 128
M.model.desc_dim = 128
M.model.fusion_type = 'max_pool'
M.model.type = 'MVPoolNet'

M.l2net = CN()
M.l2net.return_interims = False
M.l2net.trainable = True

M.view_pool = CN()
M.view_pool.bias = True
M.view_pool.kernel = 3

M.train = CN()

M.train.general = CN()
M.train.general.ckpt_nets = []
M.train.general.ckpt_path = ''

M.train.input = CN()
M.train.input.batch_size = 1
M.train.input.instance_num = 24

M.train.dataset = CN()
M.train.dataset.name = ''
M.train.dataset.pcloud_root = ''
M.train.dataset.kpts_root = ''
M.train.dataset.workers = 6

M.train.solver = CN()
M.train.solver.epochs = 16
M.train.solver.lr = 1e-3
M.train.solver.lr_gamma = 0.1
M.train.solver.lr_step = 4 
M.train.solver.optim = 'Adam' 
M.train.solver.renderer_lr = 1e-3
M.train.solver.renderer_optim_step = 10
M.train.solver.renderer_weight = 1.
M.train.solver.save_freq = 4 
M.train.solver.weight_decay = -1
M.train.solver.grad_clip = 1e-3
 
M.eval = CN()

M.eval.general = CN()
M.eval.general.ckpt_path = ''

M.eval.geomreg = CN()
M.eval.geomreg.workers = 0

M.eval.geomreg.valid = CN()
M.eval.geomreg.valid.name = ''
M.eval.geomreg.valid.pcloud_root = ''

M.eval.geomreg.test = CN()
M.eval.geomreg.test.name = ''
M.eval.geomreg.test.pcloud_root = ''
