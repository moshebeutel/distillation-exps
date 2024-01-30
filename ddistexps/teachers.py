import torch
import argparse
import os
import csv
import torch.nn as nn

import ddist.models
from ddist.models.resnet_cifar import resnet20 as resnet20cf10
from ddist.models.resnet_cifar import resnet56 as resnet56cf10
from ddist.models.clip import clip_vitb32
from torchvision.models import resnet50 as resnet50imgnet
from torchvision.models import resnet101 as resnet101imgnet
from ddist.models.densenet_cifar import densenet121, densenet169
from ddist.models.resnet_cifar import resnet56 as resnet56tin
from ddist.models.resnet_cifar import resnet110 as resnet110tin
from ddist.utils import CLog as lg
from ddistexps.utils import load_mlflow_module

class NoOpTeacher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[:, 0, 0, 0]
        frac, flr = (x/2), torch.floor(x/2)
        even = (torch.abs(frac - flr) < 0.25) * 1.0
        odd = (torch.abs(frac - flr) > 0.25) * -1.0
        even = even.to(dtype=torch.float32)
        odd = odd.to(dtype=torch.float32)
        return torch.unsqueeze(even + odd, dim=-1)
    

def get_teacher_model(teacher_cfg):
    """
    Returns an nn.module corresponding to the specified teacher config.
    Teacher config is specified by specifying
        'expname', 'runname'
    """
    if 'RangeTrunk' in str(vars(teacher_cfg)):
        return NoOpTeacher()
    trunk, run_id = load_mlflow_module(teacher_cfg)
    return trunk


def ClipCIFAR10_args(args):
    """Returns trunks and sets the model type. """
    from ddist.data.dataset import CIFAR10Dataset
    labels = CIFAR10Dataset.metadata['labels']

    kwargs = {'labels': labels, 'model_pretrain_save': 'laion2b_s34b_b79k'}
    trunk = clip_vitb32(**kwargs)
    return trunk

