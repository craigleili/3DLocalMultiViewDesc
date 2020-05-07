from __future__ import division
from __future__ import print_function

import os.path as osp
import torch


class BaseModel(object):

    def __init__(self):
        self._nets = list()
        self._net_names = list()
        self._train_flags = list()

    def __call__(self, *args):
        pass

    def register_nets(self, nets, names, train_flags):
        self._nets.extend(nets)
        self._net_names.extend(names)
        self._train_flags.extend(train_flags)

    def params(self, trainable, named=False, add_prefix=False):
        def _get_net_params(_net, _net_name):
            if named:
                if add_prefix:
                    return [(_net_name + '.' + _param_name, _param_data)
                            for _param_name, _param_data in _net.named_parameters()]
                else:
                    return list(_net.named_parameters())
            else:
                return list(_net.parameters())

        res = list()
        for idx, net in enumerate(self._nets):
            net_flag = self._train_flags[idx]
            net_name = self._net_names[idx]

            if trainable:
                if net_flag:
                    res.extend(_get_net_params(net, net_name))
            else:
                res.extend(_get_net_params(net, net_name))
        return res

    def params_to_optimize(self, l2_weight_decay, excludes=('bias',)):
        if l2_weight_decay > 0:
            if excludes is None:
                excludes = list()

            decay_params = list()
            nondecay_params = list()

            named_params = self.params(True, named=True, add_prefix=False)
            for param_name, param_data in named_params:
                use_decay = True
                for kw in excludes:
                    if kw in param_name:
                        use_decay = False
                        break
                if use_decay:
                    decay_params.append(param_data)
                else:
                    nondecay_params.append(param_data)
            return [{
                'params': decay_params,
                'weight_decay': l2_weight_decay
            }, {
                'params': nondecay_params,
                'weight_decay': 0
            }]
        else:
            return self.params(True, named=False, add_prefix=False)

    def print_params(self, model_name='Model'):
        print('[*] {} parameters:'.format(model_name))
        for nid, net in enumerate(self._nets):
            if self._train_flags[nid]:
                print('[*]   Trainable module {}'.format(self._net_names[nid]))
            else:
                print('[*]   None-trainable module {}'.format(self._net_names[nid]))
            for name, param in net.named_parameters():
                print('[*]     {}: {}'.format(name, param.size()))
        print('[*] {} size: {:.5f}M'.format(model_name, self.num_params() / 1e6))

    def num_params(self):
        return sum(p.numel() for p in self.params(False))

    def subnet_dict(self):
        return {self._net_names[i]: self._nets[i] for i in range(len(self._nets))}

    def save(self, root_folder, filename_prefix, iteration, solver_state=None):
        res = list()
        for net, name in zip(self._nets, self._net_names):
            net_path = osp.join(root_folder, '{}_{}_{}.pth'.format(filename_prefix, name,
                                                                   iteration))
            torch.save(net.state_dict(), net_path)
            res.append(net_path)
        if solver_state is not None:
            solver_state_path = osp.join(root_folder,
                                         '{}_solver_{}.pth'.format(filename_prefix, iteration))
            torch.save(solver_state, solver_state_path)
            res.append(solver_state_path)
        return res

    def load(self, path_pattern, net_names=None):
        print('[*] Load Pretrained Parameters:')

        def load_net(name, net, pth_path):
            model_dict = net.state_dict()
            pretrained_dict = torch.load(pth_path)
            print('[*]  Module {} from {}'.format(name, pth_path))
            filtered_dict = dict()
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if model_dict[k].size() == pretrained_dict[k].size():
                        filtered_dict[k] = v
                        print('[*]    Use {}, {}'.format(k, pretrained_dict[k].size()))
                    else:
                        print('[*]    Discard {}, {} and {} do not match'.format(
                            k, model_dict[k].size(), pretrained_dict[k].size()))
                else:
                    print('[*]    Discard unknown {}'.format(k))
            model_dict.update(filtered_dict)
            net.load_state_dict(model_dict)

        if net_names is None or len(net_names) == 0:
            for net, name in zip(self._nets, self._net_names):
                net_path = path_pattern.format(name)
                if osp.exists(net_path):
                    load_net(name, net, net_path)
        else:
            for net, name in zip(self._nets, self._net_names):
                if name not in net_names: 
                    continue
                net_path = path_pattern.format(name)
                if not osp.exists(net_path):
                    raise RuntimeError("[!] {} does not exist.".format(net_path))
                load_net(name, net, net_path)

    def train_mode(self):
        for net, train_flag in zip(self._nets, self._train_flags):
            if train_flag:
                net.train()
            else:
                net.eval()

    def eval_mode(self):
        for net in self._nets:
            net.eval()

    def to(self, device):
        for net in self._nets:
            net.to(device)
