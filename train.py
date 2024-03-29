# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Train.
"""
import argparse
import os

from collections import Counter
import math
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.config import imagenet_cfg
from src.dataset import create_dataset_imagenet
from src.modeling_ms import VisionTransformer
import src.configs as configs

set_seed(1)

def lr_steps_cifar10(global_step, lr_max=None, total_epochs=None, steps_per_epoch=None):
    """Set learning rate."""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr_each_step.append(lr_max)
        elif i < decay_epoch_index[1]:
            lr_each_step.append(lr_max * 0.1)
        elif i < decay_epoch_index[2]:
            lr_each_step.append(lr_max * 0.01)
        else:
            lr_each_step.append(lr_max * 0.001)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def lr_steps_imagenet(_cfg, steps_per_epoch):
    """lr step for imagenet"""
    if _cfg.lr_scheduler == 'exponential':
        _lr = warmup_step_lr(_cfg.lr_init,
                             _cfg.lr_epochs,
                             steps_per_epoch,
                             _cfg.warmup_epochs,
                             _cfg.epoch_size,
                             gamma=_cfg.lr_gamma,
                             )
    elif _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         _cfg.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr1 = float(init_lr) + lr_inc * current_step
    return lr1


def warmup_step_lr(lr2, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """warmup step lr"""
    base_lr = lr2
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr2 = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr2 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr2 = lr2 * gamma ** milestones_steps_counter[i]
        lr_each_step.append(lr2)

    return np.array(lr_each_step).astype(np.float32)


def multi_step_lr(lr3, milestones, steps_per_epoch, max_epoch, gamma=0.1):
    """lr"""
    return warmup_step_lr(lr3, milestones, steps_per_epoch, 0, max_epoch, gamma=gamma)


def step_lr(lr4, epoch_size, steps_per_epoch, max_epoch, gamma=0.1):
    """lr"""
    lr_epochs = []
    for i in range(1, max_epoch):
        if i % epoch_size == 0:
            lr_epochs.append(i)
    return multi_step_lr(lr4, lr_epochs, steps_per_epoch, max_epoch, gamma=gamma)


def warmup_cosine_annealing_lr(lr5, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """ warmup cosine annealing lr"""
    base_lr = lr5
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr5 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr5)

    return np.array(lr_each_step).astype(np.float32)


class CrossEntropySmooth(_Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss2 = self.ce(logit, label)
        return loss2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet', 'cifar10'],
                        help='dataset name.')
    parser.add_argument('--device_id', type=int, default=None, help='device id of GPU or Ascend. (Default: None)')
    args_opt = parser.parse_args()

    if args_opt.dataset_name == "imagenet":
        cfg = imagenet_cfg
    else:
        raise ValueError("Unsupported dataset.")

    # set context
    device_target = cfg.device_target

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
    context.set_context(enable_graph_kernel=True)
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    rank = 0
    if device_target == "Ascend":
        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
    elif device_target == "GPU":
        if args_opt.device_id is not None:
            context.set_context(device_id=args_opt.device_id)
        else:
            context.set_context(device_id=cfg.device_id)
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
    else:
        raise ValueError("Unsupported platform.")

    if args_opt.dataset_name == "imagenet":
        dataset = create_dataset_imagenet(cfg.data_path, 1)
    else:
        raise ValueError("Unsupported dataset.")

    batch_num = dataset.get_dataset_size()

    net = VisionTransformer(configs.get_b16_config, num_classes=cfg.num_classes)
    # Continue training if set pre_trained to be True
    if cfg.pre_trained:
        param_dict = load_checkpoint(cfg.checkpoint_path)
        load_param_into_net(net, param_dict)

    loss_scale_manager = None
    if args_opt.dataset_name == 'cifar10':
        lr = lr_steps_cifar10(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size, steps_per_epoch=batch_num)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                       learning_rate=Tensor(lr),
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    elif args_opt.dataset_name == 'imagenet':
        lr = lr_steps_imagenet(cfg, batch_num)


        def get_param_groups(network):
            """ get param groups """
            decay_params = []
            no_decay_params = []
            for x in network.trainable_params():
                parameter_name = x.name
                if parameter_name.endswith('.bias'):
                    # all bias not using weight decay
                    no_decay_params.append(x)
                elif parameter_name.endswith('.gamma'):
                    # bn weight bias not using weight decay, be carefully for now x not include BN
                    no_decay_params.append(x)
                elif parameter_name.endswith('.beta'):
                    # bn weight bias not using weight decay, be carefully for now x not include BN
                    no_decay_params.append(x)
                else:
                    decay_params.append(x)

            return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


        if cfg.is_dynamic_loss_scale:
            cfg.loss_scale = 1

        opt = Momentum(params=get_param_groups(net),
                       learning_rate=Tensor(lr),
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay,
                       loss_scale=cfg.loss_scale)
        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                  amp_level="O3", keep_batchnorm_fp32=False, loss_scale_manager=loss_scale_manager)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 2, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="train_vit_" + args_opt.dataset_name, directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()
    model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    print("train success")
