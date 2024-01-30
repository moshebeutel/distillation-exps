import numpy as np
import torch
from ddist.models import ResidualCNN, Composer
from ddistexps.utils import load_mlflow_module
from ddist.utils import namespace_to_dict
from ddistexps.composition.gen_forwards import (
    fwd_noconnection, fwd_residual, fwd_share_all,
    fwd_share_post_layer
)

from .debug import EXPERIMENTS as expdebug

ALL_VALID = [expdebug]
EXPERIMENTS = {}
for exp in ALL_VALID:
    for k in exp.keys():
        EXPERIMENTS[k] = exp[k]

CONNDICT = {
    'noconn': fwd_noconnection,
    'residual_error': fwd_residual,
    'share_all': fwd_share_all,
    'share_post_layer': fwd_share_post_layer,
}

def resnetgen(in_H=32, in_W=32, out_dim=10):
    cands = {
        'fn': ResidualCNN,
        'kwargs': [
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 1, 'out_dim': out_dim,
            'blocks_list': [1], 'emb_list': [16], 'stride_list': [1]},
            {'in_H': in_H, 'in_W': in_W, 'num_layers': 2, 'out_dim': out_dim,
            'blocks_list': [1, 1], 'emb_list': [16, 16], 'stride_list': [1, 2]},
        ],
    }
    return cands

def get_composed_model(input_cfg, module_cfg, composer_cfg):
    """
    Returns a composed model. The composition is constructed by
    connection the module with the specified pretrained model
    and connection in composer_cfg.
    """
    def dry_run(model, input_size):
        x = np.random.normal(size=(2,) + input_size)
        x = torch.tensor(x, dtype=torch.float32)
        _ = model(x)
        return model

    module_cls, module_kwargs = module_cfg.fn, module_cfg.kwargs
    module = module_cls(**namespace_to_dict(module_kwargs))
    input_size = input_cfg.input_shape
    module = dry_run(module, input_size)

    run_cfg = composer_cfg.src_run_cfg
    src_model, _ = load_mlflow_module(run_cfg)
    for n, p in src_model.named_parameters():
        p.requires_grad = False
    conname = composer_cfg.conn_name
    container = module 
    if conname is not None:
        fwd_fn = CONNDICT[conname]
        container = Composer([src_model], module, fwd_fn)
    return container

def get_candgen(meta, ds_meta):
    in_H, in_W = ds_meta['image_dims'][1:]
    outdim = ds_meta['num_labels']
    if meta['gridgen'] in ['resnetgen']:
        grid_records = resnetgen(in_H, in_W, outdim)
    else:
        raise ValueError(meta['gridgen'])
    return grid_records
