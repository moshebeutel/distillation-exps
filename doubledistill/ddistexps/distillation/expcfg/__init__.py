from doubledistill.ddist import ResidualCNN
from doubledistill.ddist import ResNetv3Patched
from .cifar10 import EXPERIMENTS as expscf10
from .cifar100 import EXPERIMENTS as expscf100
from .debug import EXPERIMENTS as expdebug

ALL_VALID = [expscf10, expscf100, expdebug]
EXPERIMENTS = {}
for exp in ALL_VALID:
    for k in exp.keys():
        EXPERIMENTS[k] = exp[k]


def resnetgen(in_H=32, in_W=32, out_dim=10):
    # Define product-space or explicit records here.
    # The configuration here is added to the experiment spec as 
    #   spec['module_cfg'] = module_grid
    # Here module_grid is returned by this function can thus be
    # any non-closured product space.
    cands = {
        'fn': ResidualCNN,
        'kwargs': [
            {'in_H': 32, 'in_W': 32, 'num_layers': 1, 'out_dim': out_dim,
            'blocks_list': [1], 'emb_list': [8], 'stride_list': [1]},
            {'in_H': 32, 'in_W': 32, 'num_layers': 1, 'out_dim': out_dim,
            'blocks_list': [2], 'emb_list': [8], 'stride_list': [1]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 2, 'out_dim': out_dim,
            # 'blocks_list': [1, 1], 'emb_list': [8,16], 'stride_list': [1, 2]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 2, 'out_dim': out_dim,
            # 'blocks_list': [1, 2], 'emb_list': [8,16], 'stride_list': [1, 2]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 2, 'out_dim': out_dim,
            # 'blocks_list': [2, 2], 'emb_list': [16,16], 'stride_list': [1, 2]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 2, 'out_dim': out_dim,
            # 'blocks_list': [2, 3], 'emb_list': [16,16], 'stride_list': [1, 2]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 3, 'out_dim': out_dim,
            # 'blocks_list': [2, 2, 2], 'emb_list': [8,16, 32], 'stride_list': [1, 2, 2]},
            # {'in_H': 32, 'in_W': 32, 'num_layers': 3, 'out_dim': out_dim,
            # 'blocks_list': [2, 3,3], 'emb_list': [16,16,32], 'stride_list': [1, 2,2]},
            # {'in_H': in_H, 'in_W': in_W, 'num_layers': 2, 'out_dim': out_dim,
            # 'blocks_list': [1, 1], 'emb_list': [8, 16], 'stride_list': [1, 2]},
            # {'in_H': in_H, 'in_W': in_W, 'num_layers': 2, 'out_dim': out_dim,
            # 'blocks_list': [1, 2], 'emb_list': [16, 32], 'stride_list': [1, 2]},
            # # Resnet 18
            # {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            # 'blocks_list': [3, 3, 3], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
            # # Resnet 
            # {'in_H': in_H, 'in_W': in_W, 'num_layers': 3, 'out_dim': out_dim,
            # 'blocks_list': [12, 12, 12], 'emb_list': [16, 32, 64], 'stride_list': [1, 2, 2]},
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
