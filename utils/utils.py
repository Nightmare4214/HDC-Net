#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import random

import numpy as np
import torch


def setup_seed(seed):
    """
    set random seed
    :param seed: seed num
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # LSTM(cuda>10.2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - epoch / MAX_EPOCHES, power), 8)