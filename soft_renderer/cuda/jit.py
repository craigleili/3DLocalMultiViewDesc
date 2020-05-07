import os.path as osp
from torch.utils.cpp_extension import load

current_folder = osp.realpath(osp.abspath(osp.dirname(__file__)))

soft_rasterize_cuda = load('soft_rasterize_cuda',
                           [current_folder + '/soft_rasterize_cuda.cpp', current_folder + '/soft_rasterize_cuda_kernel.cu'],
                           verbose=True)
