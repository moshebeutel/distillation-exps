from ddist.models import ResidualCNN
from .cifar10 import EXPERIMENTS as expscf10
from .cifar100 import EXPERIMENTS as expscf100
from .tinyimagenet import EXPERIMENTS as expstinyimagenet
from .debug import EXPERIMENTS as expdebug

ALL_VALID = [expscf10, expscf100, expdebug, expstinyimagenet]
EXPERIMENTS = {}
for exp in ALL_VALID:
    for k in exp.keys():
        EXPERIMENTS[k] = exp[k]


def resnetgen(in_H=32, in_W=32, out_dim=10):
    # Define product-space or explicit records here
    cands = {
        'fn': ResidualCNN,
        'kwargs': [
            # Resnet 18
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            'blocks_list': [3, 3, 3], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            # ResNet 36
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            'blocks_list': [6, 6, 6], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            # ResNet 96
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            'blocks_list': [12, 12, 12], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            # ResNet 144
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            'blocks_list': [24, 24, 24], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
        ],
    }
    return cands

def get_candgen(meta, ds_meta):
    in_H, in_W = ds_meta['image_dims'][1:]
    outdim = ds_meta['num_labels']
    if meta['gridgen'] in ['resnetgen']:
        grid_records = resnetgen(in_H, in_W, outdim)
    else:
        raise ValueError(meta['gridgen'])
    return grid_records
