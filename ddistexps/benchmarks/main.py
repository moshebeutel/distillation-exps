# 
# Scripts to compare various trainers against baseline trainer.
# (This script does not have a trainer of its own.)
#
import ray
import os
import time
import mlflow
import numpy as np
from argparse import ArgumentParser
from rich import print as rr

from ddist.utils import spec_to_prodspace, dict_to_namespace, namespace_to_dict
from ddistexps.baseline.trainer import BaselineTrainer as BLTrainer
from ddistexps.utils import get_dataflow
# Testing other trainers
from ddistexps.distillation.trainer import DistilTrainer
from ddistexps.distillation.teachers import NoOpTeacher
# Importing experiments
from ddistexps.benchmarks.expcfg import EXPERIMENTS
from ddistexps.benchmarks.expcfg import get_modulegrid



if __name__ == '__main__':
    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)

    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]
    # Attach sub-directory to expname
    expname = 'benchmarks/' + expname
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

    # Define all trainers we are testing and separate the payloads based on
    # these.
    distildispatch = DistilTrainer.remote(**dispatch_kwargs)
    bldispatch = BLTrainer.remote(**dispatch_kwargs)
    for p in payloads:
        _kwargs = namespace_to_dict(p.module_cfg.kwargs)
        p.module = p.module_cfg.fn(**_kwargs)
    distildispatch_payloads = [p for p in payloads if p.dispatch == 'distillation']
    bldispatch_payloads = [p for p in payloads if p.dispatch == 'baseline']
    refs = []
    if len(distildispatch_payloads) > 0:
        rr("Starting DistillationDispatch")
        trunk = NoOpTeacher()
        wlref = distildispatch.train.remote(distildispatch_payloads, trunk)
        refs = refs + [wlref]
    if len(bldispatch_payloads) > 0:
        rr("Starting BaselineDispatch")
        blref = bldispatch.train.remote(bldispatch_payloads)
        refs = refs + [blref]
    st_time = time.time()
    ray.get(refs)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)

