import numpy as np
import torch
from ddist.models import ResidualCNN, Composer
from ddist.models.resnet_cifar import ResNetv3Patched
from ddistexps.utils import load_mlflow_module
from ddist.utils import namespace_to_dict
from .debug import EXPERIMENTS as expdebug
from .cifar10 import EXPERIMENTS as expcifar10

ALL_VALID = [expdebug, expcifar10]
EXPERIMENTS = {}
for exp in ALL_VALID:
    for k in exp.keys():
        EXPERIMENTS[k] = exp[k]



def resnetgen(in_H=32, in_W=32, out_dim=10):
    cands = {
        'fn': ResidualCNN,
        'kwargs': [
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 2, 'out_dim': out_dim,
            'blocks_list': [1, 1], 'emb_list': [8, 16], 'stride_list': [1, 2]},
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 2, 'out_dim': out_dim,
            'blocks_list': [1, 1], 'emb_list': [8, 16], 'stride_list': [2, 2]},
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            'blocks_list': [1, 1, 1], 'emb_list': [8, 8, 8], 'stride_list': [2, 2, 2]},
        ],
    }
    return cands


def resnetv3patchedgen(in_H=32, in_W=32, out_dim=10):
    cands = {
        'fn': ResNetv3Patched,
        'kwargs': {'depth': [8, 14, 20], 'num_classes': out_dim}
    }
    return cands


def get_candgen(meta, ds_meta):
    in_H, in_W = ds_meta['image_dims'][1:]
    outdim = ds_meta['num_labels']
    if meta['gridgen'] in ['resnetgen']:
        grid_records = resnetgen(in_H, in_W, outdim)
    elif meta['gridgen'] in ['resnetv3patchedgen']:
        grid_records = resnetv3patchedgen(in_H, in_W, outdim)
    else:
        raise ValueError(meta['gridgen'])
    return grid_records
