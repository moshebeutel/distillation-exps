import ray
import os
import numpy as np
import hashlib
import torch
from argparse import Namespace
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

from ddist.data import DataFlowControl
from ddist.models import Composer
from ddistexps.ftdistil.dataflow import AugDataFlow
from ddist.utils import namespace_to_dict, flatten_dict
from ddist.utils import CLog as lg


def param_hash(cfg):
    cfgstr = ''.join([f'{k}={cfg[k]}\n' for k in sorted(cfg.keys())]).strip()
    cfghash = hashlib.md5(cfgstr.encode('utf-8'), usedforsecurity=False)
    cfghash = cfghash.hexdigest()
    return cfgstr, cfghash

def retrive_mlflow_run(payload, expname):
    """
    Creates a new run and returns it. If another run with the same set of
    parameters already exists, then we retain the name but add a version number
    to it.

    Returns: (is_new_run, run_id)

    version: If a run with the same set of parameters already exists, then we
        retain the name but add a version number to it.
    ignore: If a run with the same set of parameters already exists, then we
        ignore it and create a new run with a new name.

    We compare two runs by
        1. converting the namespace to a dictionary, and flattening the dictionary,
        2. We then stringify the flattened dictionary,
        3. computing the md5 hash of the string, compare.
    """
    import mlflow
    from mlflow.tracking.client import MlflowClient
    dedup_policy = payload.meta.dedup_policy
    meta = payload.meta
    del payload.meta # Don't create hash with meta
    if dedup_policy not in ['version', 'ignore']: raise ValueError(dedup_policy)
    try:
        currid = payload.mlflow_runid
    except AttributeError:
        currid = None
    if currid is not None:
        return False, currid

    exp = mlflow.set_experiment(experiment_name=expname)
    # will exclude module and trunk
    cfg = flatten_dict(namespace_to_dict(payload))
    cfgstr, cfghash = param_hash(cfg)
    query = f"params.param_hash = '{cfghash}'"
    existing_runs = MlflowClient().search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=query,
    )
    vtagsd = {r.data.tags['auto-version']: r.info.run_id for r in existing_runs}
    vtagsl = list(vtagsd.keys())
    runname = None if len(vtagsl) == 0 else existing_runs[0].data.tags['mlflow.runName']
    curr_vid = 0 if len(vtagsl) == 0 else (max([int(x) for x in vtagsl]) + 1)
    if curr_vid > 0 and dedup_policy == 'ignore':
        _v = max([int(x) for x in vtagsl])
        return False, vtagsd[str(_v)]
    _tags = {'auto-version': str(curr_vid)}
    with mlflow.start_run(run_name=runname, tags=_tags) as run:
        payload.mlflow_runid = run.info.run_id
        meta_d = {'meta.'+k:v for k, v in flatten_dict(namespace_to_dict(meta)).items()}
        cfg = cfg | meta_d
        cfg['param_hash'] = cfghash
        mlflow.log_params(cfg)
        mlflow.log_text(cfgstr, 'param_str.txt')
    return True, payload.mlflow_runid


def get_persistent_actor(actor_name, actor_class, actor_kwargs, persist=True):
    try:
        actor = ray.get_actor(actor_name)
        lg.info("[green bold] Found existing actor:", actor_name, "-->", actor)
    except ValueError:
        msg = f"[red bold] No existing actor found. "
        if persist is True:
            opts ={'lifetime': 'detached', 'name': actor_name}
        else:
            opts = {'name': actor_name}
        lg.info(msg + " Creating detached actor:", opts)
        actor_fn = actor_class.options(**opts).remote
        actor = actor_fn(**actor_kwargs)
    return actor

@ray.remote
def get_dataflow(args, world_size, engine='default'):
    # Create Dataflow
    dataflow_args = {
        'ds_name': args.data_set, 'ddp_world_size': world_size,
        'read_parallelism': args.read_parallelism,
    }
    df_actor_name = f"DF-{args.data_set}"
    enginecls = DataFlowControl
    if engine not in ['default', 'augmented']:
        raise ValueError(f"Invalid dataflow: {engine}")
    if engine in ['augmented']:
        enginecls = AugDataFlow
    dfctrl = get_persistent_actor(df_actor_name, enginecls, dataflow_args)
    return dfctrl

def load_mlflow_run_module(run):
    """Loads the latest module from mlflow registry.      
      run: A mlflow.entities.Run object
    """
    from mlflow.tracking.client import MlflowClient
    expname = MlflowClient().get_experiment(run.info.experiment_id).name
    run_cfg = {'expname': expname,
                'runname': run.info.run_name}
    return load_mlflow_module(Namespace(**run_cfg))

def load_mlflow_module(run_cfg):
    """Loads a module from mlflow registry. This function handles
      picking the right module class and type-checking arguments.
      
      run_cfg: A namespace with the following attributes:
            expname: The name of the experiment
            runname: The name of the run
    """
    import mlflow
    from mlflow.tracking.client import MlflowClient
    exp_name = run_cfg.expname
    run_name = run_cfg.runname
    exp = mlflow.get_experiment_by_name(exp_name)
    client = MlflowClient()
    existing_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f'tags.mlflow.runName = "{run_name}"'
    )
    if len(existing_runs) > 1:
        # Pick the run with max version. We assume all duplicate runs are versioned.
        version_tags = []
        for run in existing_runs:
            version = int(run.data.tags['auto-version'])
            artifacts = [f.path for f in client.list_artifacts(run.info.run_id)]
            ckpts = [f for f in artifacts if f.startswith('ep-')]
            if len(ckpts) == 0:
                version_tags.append(-1)
            else:
                version_tags.append(version)
        if len(version_tags) == 0:
            existing_runs = []
        latest_version = np.argmax(version_tags)
        existing_runs = [existing_runs[latest_version]]
    if len(existing_runs) == 0:
        raise ValueError(f"Invalid run_cfg config: {run_cfg}. No runs found.")

    run = existing_runs[0]
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
        return model, run.info.run_id
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
    trunk = model_cls(**model_kwargs)
    a, b = trunk.load_state_dict(sd, strict=True)
    return trunk, run.info.run_id


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