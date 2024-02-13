import time
import numpy as np
import mlflow
from rich import print as rr

import ray
from ray.air.util.torch_dist import TorchDistributedWorker

from ddistexps.utils import get_composed_model
from ddistexps.utils import retrive_mlflow_run 
from ddistexps.distillation.trainer import DistilMapper_, DistilWorker_


@ray.remote
class ComposeMapper(DistilMapper_):
    """Syntatic sugar for local trainer. Remote trainers don't work with
    inheritance"""
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs, worker_actor_cls=ComposeWorker)

    def composetrain(self, payloads):
        rr(f"Multi-train started with {len(payloads)} candidate payloads")
        dfctrl = self.dfctrl
        fn_args = {'dfctrl': dfctrl}
        fn = self.worker_actor_cls.composetrain
        new_work = []
        for payload in payloads:
            ret = retrive_mlflow_run(payload, payload.mlflow_expname)
            is_new_run, runid = ret
            if is_new_run is True:
                new_work.append(payload)
                continue
            rr("Run already exists. Skipping training.", runid)
        map_results = self.map_workers(new_work, fn, fn_args)
        map_results_ = [res[0] for res in map_results]
        rr("Multi-train: Finished")
        return map_results_
    
    def train(self, *args, **kwargs):
        # Precautionary override
        raise NotImplementedError("Use composetrain() instead")


@ray.remote
class ComposeWorker(TorchDistributedWorker, DistilWorker_):
    def __init__(self, name): pass

    @staticmethod
    def composetrain(rank, world_size, payload, dfctrl):
        """
        Create composed models using the specified configuration before
        invoking trainer.
        """
        p = payload
        p.module = get_composed_model(p.input_cfg, p.module_cfg,
                                      p.compose_cfg)
        return DistilWorker_.train(rank, world_size, p, dfctrl)



