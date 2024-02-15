import ray
import os
import time
import mlflow
import numpy as np
from argparse import ArgumentParser
from rich import print as rr

from ddist.data import get_dataset_metadata

from ddistexps.baseline.trainer import BaselineReducer
from ddistexps.baseline.expcfg import EXPERIMENTS
from ddistexps.baseline.expcfg import get_candgen
from ddistexps.utils import setup_experiment



    

if __name__ == '__main__':
    tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    mlflow.set_tracking_uri(tracking_uri)
    rr("[green bold] Connecting to mlflow using:", tracking_uri)

    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)
    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]

    # Set the experiment name in meta
    expname = 'baseline/' + expname
    spec['meta']['expname'] = expname
    # Populate the experiment spec with the module grid 
    dataset = spec['dataflow']['ds_name']
    ds_meta = get_dataset_metadata(dataset)
    module_grid = get_candgen(spec['meta']['gridgen'], ds_meta)
    spec['module_cfg'] = module_grid

    # Setup the experiment
    meta, payloads, dfctrl = setup_experiment(spec, expname)
    # Create map-reduce actor and dispatch the training runs.
    dispatch_kwargs = { 'dfctrl': dfctrl, 'worker_cfg': meta.worker_cfg}
    bldispatch = BaselineReducer.remote(**dispatch_kwargs)
    ref = bldispatch.train.remote(payloads)
    st_time = time.time()
    # Wait for training and logging to complete
    ray.get(ref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)


