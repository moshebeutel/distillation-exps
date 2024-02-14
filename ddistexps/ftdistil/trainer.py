import time
import numpy as np
import mlflow
from rich import print as rr

import ray
from ray.experimental.tqdm_ray import tqdm
from ray.air.util.torch_dist import TorchDistributedWorker
from ray.air.util.torch_dist import get_device as ray_get_device

from torch.nn.parallel import DistributedDataParallel as DDP

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

from ddist.utils import CLog as lg
from ddistexps.utils import (
    retrive_mlflow_run, load_mlflow_run_module,
)
from ddistexps.teachers import get_teacher_model
from ddistexps.distillation.trainer import DistilMapper_, DistilWorker_


@ray.remote
class FineTuneTrainer(DistilMapper_):

    def finetune(self, payloads):
        lg.info(f"Multi-finetune started with {len(payloads)} candidate payloads")
        dfctrl = self.dfctrl
        fn_args = {'dfctrl': dfctrl}
        fn = FineTuneWorker.finetune
        new_work = []
        for payload in payloads:
            # Setup payload.mlflow_runid
            ret = retrive_mlflow_run(payload, payload.mlflow_expname)
            is_new_run, runid = ret
            if is_new_run is True:
                new_work.append(payload)
                continue
            lg.warning("Run already exists. Skipping training.", runid)
        map_results = self.map_workers(new_work, fn, fn_args)
        map_results_ = [res[0] for res in map_results]
        lg.info("Multi-train: Finished")
        return map_results_



@ray.remote
class FineTuneWorker(DistilWorker_):

    @staticmethod
    def get_all_shifts_acc(rank, world_size, payload, dfctrl, split):
        TR = FineTuneWorker
        EASY_C_SHIFTS = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur']
        MEDIUM_C_SHIFTS = ['jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
        HARD_C_SHIFTS = ['gaussian_noise', 'glass_blur', 'impulse_noise', 'shot_noise']
        INET_C_SHIFTS = ['brightness', 'contrast', 'elastic_transform', 'fog', 
                        'frost', 'jpeg_compression', 'pixelate', 'snow']
        ALL_C_SHIFTS = EASY_C_SHIFTS + MEDIUM_C_SHIFTS + HARD_C_SHIFTS
        info = {}
        for shift in tqdm(ALL_C_SHIFTS):
            split_name = f'{split}-shift-{shift}'
            cor, tot = TR.testmodel(rank, world_size, payload, dfctrl, split_name)
            info[f"acc_split_name"] = (cor / tot) * 100.0
        return info

    @staticmethod
    def finetune(rank, world_size, payload, dfctrl):
        # Setup before training
        TR = FineTuneWorker
        mlflow.set_experiment(experiment_name=payload.mlflow_expname)
        runid = payload.mlflow_runid
        device = ray_get_device()
        # Load all modules
        run = mlflow.get_run(payload.src_run)
        model = load_mlflow_run_module(run)
        trunk = get_teacher_model(payload.trunk_cfg)
        payload.module, payload.trunk = model, trunk
        model, trunk = payload.module.to(device), payload.trunk.to(device)
        if (world_size > 1): 
            model = DDP(model)

        # Data and optimizer
        train_cfg = payload.train_cfg
        data_schema = ray.get(dfctrl.get_data_schema.remote())
        _ldrargs = {'split': 'train', 'rank': rank, 'ddp_world_size':
                    world_size, 'device': device, 'transform_cfg':
                    train_cfg.transform}
        shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
        to_opt = [x for x in model.parameters() if x.requires_grad]
        optim_cfg = train_cfg.optim
        nepochs = train_cfg.num_epochs
        optmzr = TR.get_optimizer(optim_cfg, to_opt)
        lr_sched = TR.get_lr_sched(optim_cfg, optmzr, nepochs)

        # Pre-trained accuracy
        corr, tot = TR.testmodel(rank, world_size, payload, dfctrl, 'val')
        pval_acc = (corr / tot) * 100.0
        with mlflow.start_run(run_id=runid):
            mlflow.log_metrics({'pretrained_acc': pval_acc})
        # Train loop
        start_time = time.time()
        for ep in range(train_cfg.num_epochs):
            _eps = {'device': device, 'shard': shard, 'data_schema':
                    data_schema, 'payload': payload, 'trunk': trunk,
                    'optimizer': optmzr, 'ep': ep}
            st_time = time.time()
            tr_summary = TR._epoch(**_eps)
            lr_sched.step()
            en_time = time.time()
            if (ep % 10 == 0) or (ep == train_cfg.num_epochs - 1):
                corr, tot = TR.testmodel(rank, world_size, payload, dfctrl, 'val')
                val_acc = (corr / tot) * 100.0
                corrn, totn = TR.testmodel(rank, world_size, payload, dfctrl, 'noise_val')
                tr_summary['val_acc'] = val_acc
                tr_summary['clean_acc'] = val_acc
                tr_summary['noise_acc'] = (corrn / totn) * 100.0
            tr_summary['epoch_duration'] = en_time - st_time
            if rank == 0:
                with mlflow.start_run(run_id=runid):
                    mlflow.log_metrics(tr_summary, step=ep)
        info = {'train_duration': time.time() - start_time}
        if (world_size > 1): model = model.module
        model = model.to('cpu')
        if rank == 0:
            sinfo = TR.get_all_shifts_acc(rank, world_size, payload, dfctrl, 'val')
            info = info | sinfo
            with mlflow.start_run(run_id=runid):
                mlflow.log_metrics(info, step=train_cfg.num_epochs)
                name = 'ep-' + str(train_cfg.num_epochs)
                mlflow.pytorch.log_state_dict(model.state_dict().copy(), name)
        return 
