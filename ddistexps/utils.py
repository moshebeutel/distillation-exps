import ray
import os
import numpy as np
import hashlib
import torch
from argparse import Namespace
from rich import print as rr

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

import mlflow
from mlflow.tracking.client import MlflowClient

from ddist.data import DataFlowControl
from ddist.models import Composer

from ddist.utils import (
    spec_to_prodspace, dict_to_namespace, namespace_to_dict,
    flatten_dict
)


def get_persistent_actor(actor_name, actor_class, actor_kwargs, persist=True):
    """Returns a ray actor in persistent mode. If the actor already exists
    then it returns the existing actor. If the actor does not exist, then it
    creates a new actor with the specified name and class."""
    try:
        actor = ray.get_actor(actor_name)
        rr("[green bold] Found existing actor:", actor_name, "-->", actor)
    except ValueError:
        msg = f"[red bold] No existing actor found. "
        if persist is True:
            opts ={'lifetime': 'detached', 'name': actor_name}
        else:
            opts = {'name': actor_name}
        rr(msg + " Creating detached actor:", opts)
        actor_fn = actor_class.options(**opts).remote
        actor = actor_fn(**actor_kwargs)
    return actor

@ray.remote
def get_dataflow(df_kwargs, engine='default'):
    # Create Dataflow
    df_actor_name = f"DF-{df_kwargs['ds_name']}"
    enginecls = DataFlowControl
    if engine not in ['default', 'augmented']:
        raise ValueError(f"Invalid dataflow: {engine}")
    if engine in ['augmented']:
        enginecls = AugDataFlow
    dfctrl = get_persistent_actor(df_actor_name, enginecls, df_kwargs)
    return dfctrl

def param_hash(cfg):
    cfgstr = ''.join([f'{k}={cfg[k]}\n' for k in sorted(cfg.keys())]).strip()
    cfghash = hashlib.md5(cfgstr.encode('utf-8'), usedforsecurity=False)
    cfghash = cfghash.hexdigest()
    return cfgstr, cfghash


def get_existing_runs_in_exp(run_cfg, expname):
    """
    Returns a list of runs with the same parameters as the run_cfg in the
    current experiment.

    We compare two runs by
        1. converting the namespace to a dictionary, and flattening the dictionary,
        2. We then stringify the flattened dictionary,
        3. computing the md5 hash of the string, compare.

    All parameters except the 'meta' are used to compute the hash.
    """
    # Creates the experiment if it does not exist
    exp = mlflow.set_experiment(experiment_name=expname)
    # Remove the meta parameteres
    run_ = namespace_to_dict(run_cfg)
    del run_['meta']
    cfg = flatten_dict(run_)
    cfgstr, cfghash = param_hash(cfg)
    query = f"params.param_hash = '{cfghash}'"
    existing_runs = MlflowClient().search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=query,
    )
    for run in existing_runs:
        # We need to maintain a param_hash for all_runs.
        # This means deduplicate_runs was not called.
        assert run.data.params['param_hash']
    return existing_runs, cfghash

def deduplicate_runs(run_cfg_list, expname, policy='skip'):
    if policy not in ['recreate', 'skip', 'continue']:
        raise ValueError(f"Invalid policy: {policy}")
    deduped = []
    for run_cfg in run_cfg_list:
        existing_runs, cfghash = get_existing_runs_in_exp(run_cfg, expname)
        if len(existing_runs) > 0 and policy == 'skip':
            # rr("[yellow] Skipping existing run:", cfghash)
            continue
        # Either existing_runs == 0, or policy in ['createnew', 'continue']
        # The trainer has to handle continue. We add the cfghash
        run_cfg.param_hash = cfghash
        deduped.append(run_cfg)
    return deduped

def setup_experiment(spec, expname, dfnamespace='DataFlow'):
    """
    Given a spec from expcfg/, this function sets up the experiment by
        1. Creating the product space of parameters
        2. Applying the deduplication policy.
        3. Create the dataflow
        4. Connect to mlflow and ray.

        Returns: run_cfg_list, dfctrl
    """
    ray.init(namespace=dfnamespace)
    meta = dict_to_namespace(spec['meta'])
    
    dfargs = spec['dataflow']
    dfargs['ddp_world_size'] = meta.worker_cfg.world_size
    dfctrl = ray.get(get_dataflow.remote(dfargs))
    rr("DF Actor ready:", ray.get(dfctrl.ready.remote()))

    prod_space = spec_to_prodspace(**spec)
    run_cfgs = [dict_to_namespace(p) for p in prod_space]
    newruns = deduplicate_runs(run_cfgs, expname, meta.dedup_policy) 
    rr(f"Deduplication policy:{meta.dedup_policy}: {len(run_cfgs)} --> {len(newruns)}")
    return meta, newruns, dfctrl


def load_mlflow_module(runid):
    run = MlflowClient().get_run(runid)
    return load_mlflow_run_module(run)

def load_mlflow_run_module(run):
    """Loads the latest module from mlflow artifacts.      
      run: A mlflow.entities.Run object
    """
    client = MlflowClient()
    artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
    ckpts = [f for f in artifacts if f.startswith('ep-')]
    if len(ckpts) == 0:
        raise ValueError(f"No ckpts in run-cfg: {run_cfg} Artifacts: {artifacts}") 
    # Get last epoch checkpoint
    epochs = [int(f[len("ep-"):]) for f in ckpts]
    ckpt = 'ep-' + str(max(epochs))
    artifact_uri = str(run.info.artifact_uri) + '/' + ckpt
    artifact_files = os.listdir(artifact_uri)
    if 'state_dict.pth' not in artifact_files:
        # Not a state dict, we will try to load the model directly
        model = mlflow.pytorch.load_model(artifact_uri)
        return model
    # If its a state-dict its hard to figure out container structure. We can only
    # do this for simples cases.
    sd = mlflow.pytorch.load_state_dict(artifact_uri)
    model_cls = run.data.params['module_cfg.fn']
    # Get kwargs
    kwarg_keys = [k for k in run.data.params.keys() if k.startswith('module_cfg.kwargs')]
    model_kwargs = {k.split('.')[-1]: run.data.params[k] for k in kwarg_keys}
    if 'ddist.models.ResidualCNN' in model_cls:
        from ddist.models import ResidualCNN
        model_cls = ResidualCNN
        model_kwargs = {k: eval(val) for k, val in model_kwargs.items()}
    elif 'ddist.models.resnet_cifar.ResNetv3Patched' in model_cls:
        from ddist.models.resnet_cifar import ResNetv3Patched
        model_cls = ResNetv3Patched
        model_kwargs = {k: eval(val) for k, val in model_kwargs.items()}
    elif 'ddist.models.resnet_cifar.ResNetv3' in model_cls:
        from ddist.models.resnet_cifar import ResNetv3
        model_cls = ResNetv3
        model_kwargs = {k: eval(val) for k, val in model_kwargs.items()}
    elif 'ddist.models.resnet_cifar.BasicResNetv2' in model_cls:
        from ddist.models.resnet_cifarv1 import BasicResNetv2
        model_cls = BasicResNetv2
        model_kwargs = {k: eval(val) for k, val in model_kwargs.items()}
    else:
        raise ValueError("Unknown trunk:", model_cls)
    model = model_cls(**model_kwargs)
    a, b = model.load_state_dict(sd, strict=True)
    return model


def profile_module(device, mdl_or_mdl_list, inp_shape, stats='flops', batched_inp_shape=False):
    mdl_list = mdl_or_mdl_list
    if type(mdl_list) is not list: mdl_list = [mdl_list]

    if batched_inp_shape is False:
        inp_shape = (1,) + inp_shape
    
    x = torch.tensor(np.ones(inp_shape)).float()
    x = x.to(device)
    # Execute once to make sure everything is initialized
    flops_list = []
    for model in mdl_list:
        model.to(device)
        # Execute once to initialize lazy layers and such
        loss = model(x)
        prof = FlopsProfiler(model)
        prof.start_profile()
        loss = model(x)
        flops = prof.get_total_flops()
        prof.end_profile()
        flops_list.append(flops)
    return flops_list

# Connections 
from ddistexps.composition.gen_forwards import (
    fwd_noconnection, fwd_residual, fwd_share_all,
    fwd_share_post_layer, fwd_layer_by_layer, fwd_resnetconn,
)

CONNDICT = {
    'noconn': fwd_noconnection,
    'residual_error': fwd_residual,
    'share_all': fwd_share_all,
    'share_post_layer': fwd_share_post_layer,
    'resnet_conn': fwd_resnetconn,
    'layer_by_layer': fwd_layer_by_layer,
}

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

    runid = composer_cfg.src_run_id
    run = MlflowClient().get_run(runid)
    src_model = load_mlflow_run_module(run)
    for n, p in src_model.named_parameters():
        p.requires_grad = False
    conname = composer_cfg.conn_name
    container = module 
    if conname is not None:
        fwd_fn = CONNDICT[conname]
        container = Composer([src_model], module, fwd_fn)
    return container