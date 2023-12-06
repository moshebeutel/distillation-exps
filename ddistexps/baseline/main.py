import ray
import os
import time
import mlflow
import numpy as np
from argparse import ArgumentParser
from rich import print as rr

from ddist.data import get_dataset
from ddist.utils import spec_to_prodspace, dict_to_namespace, namespace_to_dict

from ddistexps.baseline.trainer import BaselineTrainer as BLTrainer
from ddistexps.baseline.expcfg import EXPERIMENTS
from ddistexps.baseline.expcfg import get_candgen
from ddistexps.utils import get_dataflow


if __name__ == '__main__':
    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)

    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]
    expname = 'baseline/' + expname
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

    # Create trainer + dispatcher
    bldispatch = BLTrainer.remote(**dispatch_kwargs)
    # Attach a model to each payload
    for p in payloads:
        _kwargs = namespace_to_dict(p.module_cfg.kwargs)
        p.module = p.module_cfg.fn(**_kwargs)
    ref = bldispatch.train.remote(payloads)
    st_time = time.time()
    # Wait for training and logging to complete
    ray.get(ref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)
