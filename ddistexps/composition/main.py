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

from ddist.data import get_dataset
from ddist.utils import spec_to_prodspace, dict_to_namespace, namespace_to_dict

from ddistexps.utils import get_dataflow, get_composed_model
from ddistexps.teachers import get_teacher_model
from ddistexps.composition.expcfg import get_candgen, EXPERIMENTS
from ddistexps.composition.trainer import ComposeTrainer


if __name__ == '__main__':
    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)

    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]
    # Attach sub-directory to expname
    expname = 'composition/' + expname
    spec['mlflow_expname'] = [expname]
    meta = spec['meta']

    dataset = spec['dataflow']['data_set']
    ds_meta = get_dataset(dataset).metadata
    module_grid = get_candgen(meta, ds_meta)
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

    for p in payloads:
        p.trunk = get_teacher_model(p.trunk_cfg)
        p.module = get_composed_model(p.input_cfg, p.module_cfg,
                                      p.compose_cfg)
    cdispatch = ComposeTrainer.remote(**dispatch_kwargs)
    resultsref = cdispatch.train.remote(payloads)
    st_time = time.time()
    ray.get(resultsref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)

