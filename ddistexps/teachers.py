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
    
def _load_mlflow_module(model_cls, model_kwargs, sd):
    """Loads a model from mlflow registry. This function handles
      picking the right module class and type-checking arguments."""
    if 'ddist.models.ResidualCNN' in model_cls:
        model_cls = ddist.models.ResidualCNN
        model_kwargs = {k: eval(val) for k, val in model_kwargs.items()}
    else:
        raise ValueError("Unknown trunk:", model_cls)
    trunk = model_cls(**model_kwargs)
    a, b = trunk.load_state_dict(sd, strict=True)
    return trunk

def get_teacher_model(teacher_cfg):
    """
    Returns an nn.module corresponding to the specified teacher config.
    Teacher config is specified by specifying
        'experiment-name', 'run-name'
    """
    if 'RangeTrunk' in str(vars(teacher_cfg)):
        return NoOpTeacher()
    import mlflow
    from mlflow.tracking.client import MlflowClient
    exp_name = teacher_cfg.expname
    run_name = teacher_cfg.runname
    exp = mlflow.get_experiment_by_name(exp_name)
    client = MlflowClient()
    existing_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}"'
    )
    if len(existing_runs) == 0:
        raise ValueError(f"Invalid teacher config: {teacher_cfg}. No runs found.")
    if len(existing_runs) > 1:
        raise ValueError(f"Invalid teacher config: {teacher_cfg}. Multiple runs found.")
    run = existing_runs[0]
    artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
    if len(artifacts) == 0:
        raise ValueError(f"Invalid teacher config: {teacher_cfg}. No artifacts found.") 
    ckpt = [f for f in artifacts if f.startswith('ep-')][-1]
    artifact_uri = str(run.info.artifact_uri) + '/' + ckpt
    sd = mlflow.pytorch.load_state_dict(artifact_uri)
    model_cls = run.data.params['module_cfg.fn']
    # Get kwargs
    kwarg_keys = [k for k in run.data.params.keys() if k.startswith('module_cfg.kwargs')]
    model_kwargs = {k.split('.')[-1]: run.data.params[k] for k in kwarg_keys}
    return _load_mlflow_module(model_cls, model_kwargs, sd)



def ClipCIFAR10_args(args):
    """Returns trunks and sets the model type. """
    from ddist.data.dataset import CIFAR10Dataset
    labels = CIFAR10Dataset.metadata['labels']

    kwargs = {'labels': labels, 'model_pretrain_save': 'laion2b_s34b_b79k'}
    trunk = clip_vitb32(**kwargs)
    return trunk

