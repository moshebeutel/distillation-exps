import torch
import argparse
import os
import csv
import torch.nn as nn

import ddist.models
from ddist.models.clip import clip_vitb32
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
    elif 'ClipCIFAR10' in str(vars(teacher_cfg)):
        return ClipCIFAR10_args()
    trunk, run_id = load_mlflow_module(teacher_cfg)
    return trunk


def ClipCIFAR10_args():
    """Returns trunks and sets the model type. """
    from ddist.data.dataset import CIFAR10Dataset
    labels = CIFAR10Dataset.metadata['labels']

    kwargs = {'labels': labels, 'model_pretrain_save': 'laion2b_s34b_b79k'}
    trunk = clip_vitb32(**kwargs)
    return trunk

