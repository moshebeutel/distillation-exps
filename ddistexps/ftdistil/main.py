# 
#  Standard temperature based distillation
#
import ray
import os
import time
import mlflow
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from rich import print as rr

from ddist.utils import spec_to_prodspace, dict_to_namespace, namespace_to_dict, flatten_dict
from ddistexps.utils import get_dataflow

from ddistexps.ftdistil.trainer import FineTuneTrainer
from ddistexps.ftdistil.teachers import Trunks
from ddistexps.ftdistil.expcfg import EXPERIMENTS
from ddistexps.ftdistil.expcfg import get_modulegrid

def pick_valid_artifact_uri(source_payload, allrun_df):
    from mlflow.tracking.client import MlflowClient
    source_dict = namespace_to_dict(source_payload.module_cfg.kwargs)
    sdict = {'module_cfg.kwargs.'+k: v for k,v in source_dict.items()}
    msk = None
    def is_equal(val1, val2):
        val2 = str(val2)
        return val1 == val2

    for k, v in sdict.items():
        if msk is None:
            msk = allrun_df[k].map(lambda x: is_equal(x, v))
        else:
            msk2 = allrun_df[k].map(lambda x: is_equal(x, v))
            msk = msk & msk2
        if np.sum(msk) == 0:
            raise ValueError(f"No valid runs found for {k}={v}")
    df = allrun_df[msk]
    df = df.sort_values(by='val_acc', ascending=False)
    # Inspect each run and pick first that has a valid state_dict
    artifact_uri = None
    for runid in df.run_id:
        run = mlflow.get_run(runid)
        exp = mlflow.get_experiment(run.info.experiment_id)
        artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
        if len(artifacts) == 0:
            continue
        ckpt = [f for f in artifacts if f.startswith('ep-')][-1]
        artifact_uri = str(run.info.artifact_uri) + '/' + ckpt
        sd = mlflow.pytorch.load_state_dict(artifact_uri)
        break
    if artifact_uri is None:
        raise ValueError("No artifacts could be downloaded")
    return artifact_uri

def attach_pretrained_runs(payloads):
    # Get all possible runs from mlflow registry
    SOURCE_EXPS = ['baseline/cifar10', 'distillation/distil-cifar10']
    from mlflow.tracking.client import MlflowClient
    client = MlflowClient()
    exp_ids = []
    for expname in SOURCE_EXPS:
        exp = client.get_experiment_by_name(expname)
        if exp is not None:
            exp_ids.append(exp.experiment_id)
        else:
            raise ValueError(f"Experiment {expname} not found")
    runs = client.search_runs(experiment_ids=exp_ids)
    # Create a dataframe with module_kwargs, run_id and val_acc
    kwargs_keys = [k for k in runs[0].data.params.keys() if k.startswith('module_cfg.kwargs')]
    dfdict =  {k: [r.data.params[k] for r in runs] for k in kwargs_keys} | {
        'run_id': [r.info.run_id for r in runs],
        'val_acc': [r.data.metrics['val_acc'] for r in runs],
    }
    df = pd.DataFrame(dfdict)
    for pld in payloads:
        artifact_uri = pick_valid_artifact_uri(pld, df)
        pld.pretrained_artifact_uri = artifact_uri
    return payloads

if __name__ == '__main__':
    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)

    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]
    # Attach sub-directory to expname
    expname = 'ftdistil/' + expname
    spec['mlflow_expname'] = [expname]
    meta = spec['meta']
    module_grid = get_modulegrid(meta, 100 if 'cifar100' in expname else 10)
    spec['module_cfg'] = module_grid
    prod_space = spec_to_prodspace(**spec)
    payloads = [dict_to_namespace(p) for p in prod_space]
    meta = dict_to_namespace(meta)
    # Attach the runid for the pretrained model to use for each payload.
    payloads = attach_pretrained_runs(payloads)
    # Set environment variable RAY_ADDRESS to use a pre-existing cluster
    dfnamespace = 'FDistilDataFlow'
    ray.init(namespace=dfnamespace)
    tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    mlflow.set_tracking_uri(tracking_uri)
    rr("[green bold] Connecting to mlflow using:", tracking_uri)

    dflow = prod_space[0]['dataflow']
    ref = get_dataflow.remote(dflow, meta.worker_cfg.world_size,
                              engine='augmented')
    dfctrl = ray.get(ref)
    rr("DF Actor ready:", ray.get(dfctrl.ready.remote()))
    dispatch_kwargs = { 'dfctrl': dfctrl, 'worker_cfg': meta.worker_cfg}

    dispatch = FineTuneTrainer.remote(**dispatch_kwargs)
    trunks = Trunks()
    for p in payloads:
        _kwargs = namespace_to_dict(p.module_cfg.kwargs)
        p.module = p.module_cfg.fn(**_kwargs)
        p.trunk = trunks(p.trunk_cfg)
    finetune_payloads = [p for p in payloads if p.dispatch == 'finetune']
    rr("Starting Finetune-Dispatch")
    wlref = dispatch.finetune.remote(finetune_payloads)
    st_time = time.time()
    ray.get(wlref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)

