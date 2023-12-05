import ray
import ray
from argparse import Namespace
from ddist.data import DataFlowControl
from ddistexps.ftdistil.dataflow import AugDataFlow
from ddist.utils import namespace_to_dict, flatten_dict
from ddist.utils import CLog as lg
import numpy as np
import hashlib

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
        (We don't handle hash-collisions)
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
