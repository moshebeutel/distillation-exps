# Foward functions compatible with ResNet models for
# composition experiments.
import torch
import numpy as np
import torch.nn.functional as F

def accum_ab(a, b):
    """
    Returns (a+b) truncated to the shape of a.
    """
    assert len(a.shape) == len(b.shape)
    assert a.shape[0] == b.shape[0]
    # Stack the shapes and find the common shape
    dlist = np.min(np.stack((a.shape, b.shape)), axis=0)
    stpl = tuple([slice(0, end) for end in dlist])
    # Create a zero variable for accumulation
    accum = torch.zeros(a.shape).to(a.device)
    accum[:] = a[:]
    accum[stpl] = (a[stpl] + b[stpl])
    return accum


def fwd_noconnection(self, x):
    "No reuse, just the latest model. Latest model is stored in trainable"
    return self.trainable(x)

def fwd_residual(self, x):
    """We think of the prediction as Ensemble(x) + residual(x) where the
    trainable model will be used to approximate the residual"""
    out = 0.0
    return self.ensemble(x) + self.trainable(x)

def fwd_share_all(self, x):
    """Accumulate the output of the latest ensemble member and the trainable
    model after each common layer."""
    mdl = self.trainable
    en_mdl = self.ensemble.model_list[-1]
    lbA, lbB = mdl.layer_block, en_mdl.layer_block
    n_common = min(len(lbA), len(lbB))
    outsA, outsB = mdl.pre_block(x), en_mdl.pre_block(x)
    # Share for common layers
    for i in range(n_common):
        outsA, outsB = lbA[i](outsA), lbB[i](outsB)
        outsA = accum_ab(outsA, outsB) 
    # Complete the circuit
    outsA = lbA[n_common:](outsA)
    outsA = mdl.post_block(outsA)
    return outsA

def fwd_share_post_layer(self, x):
    """Accumulate and average the last member of layer-block of the lastest and
    trainable."""
    mdl = self.trainable
    en_mdl = self.ensemble.model_list[-1]
    lbA, lbB = mdl.layer_block, en_mdl.layer_block
    outsA, outsB = mdl.pre_block(x), en_mdl.pre_block(x)
    outsA, outsB = lbA(outsA), lbB(outsB)
    outsA = accum_ab(outsA, outsB)
    return  mdl.post_block(outsA)
    
