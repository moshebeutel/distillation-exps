from pathlib import Path
import ray
import os
import time
import mlflow
from argparse import ArgumentParser
from rich import print as rr
from dotenv import load_dotenv

load_dotenv()


if __name__ == '__main__':
    from doubledistill.ddist.data import get_dataset_metadata
    from doubledistill.ddistexps.baseline.trainer import BaselineReducer
    from doubledistill.ddistexps.baseline.expcfg import EXPERIMENTS
    from doubledistill.ddistexps.baseline.expcfg import get_candgen
    from doubledistill.ddistexps.utils import setup_experiment
    assert os.environ['TORCH_DATA_DIR'] == f'{Path.home()}/datasets/', \
        f'Environment variable TORCH_DATA_DIR must be set to {Path.home()}/datasets/. Got {os.environ["TORCH_DATA_DIR"]}'
    assert os.environ['DDIST_EXPS_DIR'] == f'{Path.home()}/GIT/distillation-exps', \
        (f'Environment variable DDIST_EXPS_DIR must be set to {Path.home()}/GIT/distillation-exps.'
         f' Got {os.environ["DDIST_EXPS_DIR"]}')
    assert os.environ['MLFLOW_TRACKING_URI'] == '127.0.0.1:8080', (f'Environment variable MLFLOW_TRACKING_URI is set to'
                                                                   f' {os.environ["MLFLOW_TRACKING_URI"]}')

    tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    mlflow.set_tracking_uri(tracking_uri)
    rr("[green bold] Connecting to mlflow using:", tracking_uri)

    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default='debug')
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
    dispatch_kwargs = {'dfctrl': dfctrl, 'worker_cfg': meta.worker_cfg}
    bldispatch = BaselineReducer.remote(**dispatch_kwargs)
    ref = bldispatch.train.remote(payloads)
    st_time = time.time()
    # Wait for training and logging to complete
    ray.get(ref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
             'total-duration': time.time() - st_time}
    rr("Experiment completed:", info_)
