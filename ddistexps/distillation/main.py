# 
#  Standard temperature based distillation
#
import ray
import os
import time
import mlflow
import numpy as np
from argparse import ArgumentParser
from rich import print as rr

from ddist.utils import spec_to_prodspace, dict_to_namespace, namespace_to_dict
from ddistexps.utils import get_dataflow

from ddistexps.distillation.trainer import DistilTrainer
from ddistexps.distillation.expcfg import EXPERIMENTS
from ddistexps.distillation.expcfg import get_modulegrid
from ddistexps.distillation.teachers import get_teacher_model


if __name__ == '__main__':
    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)

    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]
    # Attach sub-directory to expname
    expname = 'distillation/' + expname
    spec['mlflow_expname'] = [expname]
    meta = spec['meta']
    module_grid = get_modulegrid(meta, 100 if 'cifar100' in expname else 10)
    spec['module_cfg'] = module_grid
    prod_space = spec_to_prodspace(**spec)
    payloads = [dict_to_namespace(p) for p in prod_space]
    meta = dict_to_namespace(meta)

    # Set environment variable RAY_ADDRESS to use a pre-existing cluster
    dfnamespace = 'DataFlow'
    ray.init(namespace=dfnamespace)
    tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    mlflow.set_tracking_uri(tracking_uri)
    rr("[green bold] Connecting to mlflow using:", tracking_uri)

    dflow = prod_space[0]['dataflow']
    dfctrl = ray.get(get_dataflow.remote(dflow, meta.worker_cfg.world_size))
    rr("DF Actor ready:", ray.get(dfctrl.ready.remote()))
    dispatch_kwargs = { 'dfctrl': dfctrl, 'worker_cfg': meta.worker_cfg}

    distildispatch_payloads = [p for p in payloads if p.dispatch == 'distillation']
    distildispatch = None
    distildispatch = DistilTrainer.remote(**dispatch_kwargs)

    for p in payloads:
        _kwargs = namespace_to_dict(p.module_cfg.kwargs)
        p.module = p.module_cfg.fn(**_kwargs)
        p.trunk = get_teacher_model(p.trunk_cfg)
    distilref = distildispatch.train.remote(distildispatch_payloads)
    st_time = time.time()
    ray.get(distilref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)

