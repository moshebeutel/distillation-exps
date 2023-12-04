from ddist.models import ResidualCNN
from .cifar10 import EXPERIMENTS as expscf10
from .cifar100 import EXPERIMENTS as expscf100
from .debug import EXPERIMENTS as expdebug

ALL_VALID = [expscf10, expscf100, expdebug]
EXPERIMENTS = {}
for exp in ALL_VALID:
    for k in exp.keys():
        EXPERIMENTS[k] = exp[k]


def resnetgen(out_dim=10):
    # Define product-space or explicit records here
    candscf10 = {
        'fn': ResidualCNN,
        'kwargs': [
            {'in_H': 32, 'in_W': 32, 'num_layers': 4, 'out_dim': out_dim,
            'blocks_list': [2, 2, 2, 2], 'emb_list': [16, 32, 32, 64], 'stride_list': [1, 2, 2, 2]},
            {'in_H': 32, 'in_W': 32, 'num_layers': 3, 'out_dim': out_dim,
            'blocks_list': [3, 3, 3], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 3, 'out_dim': out_dim,
            # 'blocks_list': [5, 5, 5], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
        ],
    }
    if out_dim == 10:
        cands = candscf10
    elif out_dim == 100:
        raise NotImplementedError(out_dim)
    else:
        raise ValueError(out_dim)
    return cands

def get_modulegrid(meta, outdim):
    if meta['gridgen'] in ['resnetgen']:
        grid_records = resnetgen(outdim)
    else:
        raise ValueError(meta['gridgen'])
    return grid_records
