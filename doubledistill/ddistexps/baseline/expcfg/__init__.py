from doubledistill.ddist.models.resnet_cifar import (
    ResNetOriginal, ResNetDefault, ResNet
)
from doubledistill.ddist.utils import spec_to_prodspace

from .cifar10 import EXPERIMENTS as expscf10
from .cifar100 import EXPERIMENTS as expscf100
from .tinyimagenet import EXPERIMENTS as expstinyimagenet
from .debug import EXPERIMENTS as expdebug

ALL_VALID = [expscf10, expscf100, expdebug, expstinyimagenet]
EXPERIMENTS = {}
for exp in ALL_VALID:
    for k in exp.keys():
        EXPERIMENTS[k] = exp[k]


def resnetdebug(out_dim=10):
    cands1 = {
        'fn': ResNetOriginal,
        'kwargs': {'depth': [20], 'num_classes': out_dim}
    }
    cands2 = {
        'fn': ResNetDefault,
        'kwargs': {'depth': [20], 'num_classes': out_dim}
    }
    cands3 = {
        'fn': ResNet,
        'kwargs': [
            {'blocks_list': [3, 3, 3], 'emb_list': [16, 32, 64],
             'stride_list': [1, 2, 2], 'num_classes': out_dim},
        ],
    }
    cands = []
    vd = 0
    cands = spec_to_prodspace(verbose_depth=vd, **cands1)
    cands = cands + spec_to_prodspace(verbose_depth=vd, **cands2)
    cands = cands + spec_to_prodspace(verbose_depth=vd, **cands3)
    return cands

def resnetsmall(spec_to_prodspace, out_dim=10):
    from doubledistill.ddist.utils import spec_to_prodspace
    cands1 = {
        'fn': ResNetOriginal,
        'kwargs': {'depth': [8, 14], 'num_classes': out_dim}
    }
    cands2 = {
        'fn': ResNetDefault,
        'kwargs': {'depth': [8, 14], 'num_classes': out_dim}
    }
    cands3 = {
        'fn': ResNet,
        'kwargs': [
            # # 8
            # {'num_classes': out_dim,
            #   'blocks_list': [1, 1, 1], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            # # 14
            # {'num_classes': out_dim,
            #   'blocks_list': [2, 2, 2], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},

            # General Small models
            {'num_classes': out_dim,
              'blocks_list': [1], 'emb_list': [8], 'stride_list': [1]},
            {'num_classes': out_dim,
              'blocks_list': [1], 'emb_list': [8], 'stride_list': [2]},
            {'num_classes': out_dim,
              'blocks_list': [1], 'emb_list': [16], 'stride_list': [2]},
            {'num_classes': out_dim,
              'blocks_list': [1, 1], 'emb_list': [8, 16], 'stride_list': [1, 2]},
            {'num_classes': out_dim,
              'blocks_list': [1, 1], 'emb_list': [8, 16], 'stride_list': [2, 2]},
            {'num_classes': out_dim,
              'blocks_list': [1, 1], 'emb_list': [8, 8], 'stride_list': [2, 2]},
        ],
    }
    cands = []
    vd = 0
    # cands = spec_to_prodspace(verbose_depth=vd, **cands1)
    # cands = cands + spec_to_prodspace(verbose_depth=vd, **cands2)
    cands = cands + spec_to_prodspace(verbose_depth=vd, **cands3)
    return cands


def resnetlarge(out_dim=10):
    from doubledistill.ddist.utils import spec_to_prodspace
    cands1 = {
        'fn': ResNetOriginal,
        'kwargs': {'depth': [20, 32, 44, 110], 'num_classes': out_dim}
    }
    cands2 = {
        'fn': ResNetDefault,
        'kwargs': {'depth': [20, 32, 44, 110], 'num_classes': out_dim}
    }
    cands3 = {
        'fn': ResNet,
        'kwargs': [
            {'num_classes': out_dim,
              'blocks_list': [3, 3, 3], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            {'num_classes': out_dim,
              'blocks_list': [10, 10, 10], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            {'num_classes': out_dim,
              'blocks_list': [14, 14, 14], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            {'num_classes': out_dim,
              'blocks_list': [36, 36, 36], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
        ],
    }
    cands = []
    vd = 0
    cands = spec_to_prodspace(verbose_depth=vd, **cands1)
    cands = cands + spec_to_prodspace(verbose_depth=vd, **cands2)
    cands = cands + spec_to_prodspace(verbose_depth=vd, **cands3)
    return cands


def get_candgen(gridgen, ds_meta):
    # in_H, in_W = ds_meta['image_dims'][1:]
    outdim = ds_meta['num_labels']
    if gridgen in ['resnetsmall']:
        grid_records = resnetsmall(outdim)
    elif gridgen in ['resnetlarge']:
        grid_records = resnetlarge(outdim)
    elif gridgen in ['resnetdebug']:
        grid_records = resnetdebug(outdim)
    else:
        raise ValueError(gridgen)
    return grid_records
