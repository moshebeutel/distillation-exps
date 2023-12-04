import time
import numpy as np
import mlflow
from rich import print as rr

import ray
from ray.experimental.tqdm_ray import tqdm
from ray.air.util.torch_dist import TorchDistributedWorker
from ray.air.util.torch_dist import get_device as ray_get_device

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

from ddist.utils import CLog as lg
from ddist.dispatch import BaseMapper
from ddistexps.utils import retrive_mlflow_run 


@ray.remote
class FineTuneTrainer(BaseMapper):
    """
    Implements reduce ops on the mapper outputs.
    """
    def __init__(self, dfctrl, worker_cfg):
        self.dfctrl = dfctrl
        super().__init__(_FineTuneTrainer, worker_cfg.resource_req,
                         worker_cfg.world_size, worker_cfg.num_workers)

    def finetune(self, payloads):
        lg.info(f"Multi-finetune started with {len(payloads)} candidate payloads")
        dfctrl = self.dfctrl
        fn_args = {'dfctrl': dfctrl}
        fn = _FineTuneTrainer.finetune
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

    def testmodel(self, payloads, split='train'):
        dfctrl = self.dfctrl
        fn_args = {'split': split, 'dfctrl': dfctrl}
        fn = _DistilTrainer.testmodel
        map_results = self.map_workers(payloads, fn, fn_args)
        # since each worker returns (num_correct, num_total) for each model
        # we need to do a sum-reduce
        reduce_results_ = []
        for vals in map_results:
            # Reference to (num_correct, num_total) in that shard.
            correct_samples = np.sum([elem[0] for elem in vals])
            total_samples = np.sum([elem[1] for elem in vals])
            acc = (correct_samples / total_samples) * 100.0
            reduce_results_.append(acc)
        return reduce_results_


@ray.remote
class _FineTuneTrainer(TorchDistributedWorker):
    def __init__(self, name):
        self.name = name
        lg.info("Starting worker: ", name)

    @staticmethod
    def get_model_sd(payload):
        from mlflow.tracking.client import MlflowClient
        # Get runs where the param_hash mathes.
        client = MlflowClient()
        expname = payload.mlflow_expname
        runs = client.search_runs(expname)
        param_hash = payload.param_hash
        runs = [r for r in runs if r.data.params['param_hash'] == param_hash]
        # Get the latest run.
        runs = sorted(runs, key=lambda x: x.info.start_time, reverse=True)
        run = runs[0]
        # Get the model from that run using mlflow.pytorch.load_state_dict
        sd = mlflow.pytorch.load_state_dict(run.info.run_id)
        return sd

    @staticmethod
    def ready(): return True

    @staticmethod
    def get_tempr(train_cfg, ep):
        if hasattr(train_cfg, 'temperature') is False:
            return 1.0
        nepochs = train_cfg.num_epochs
        temprcfg = train_cfg.temperature
        i  = 0
        tempr_milestones = [int(v * nepochs) for v in temprcfg.milestone_fracs]
        while (i < len(tempr_milestones)) and (ep >= tempr_milestones[i]):
            i += 1
        return temprcfg.temperature.value * (temprcfg.tempr_gamma ** i)

    @staticmethod
    def get_optimizer(optim_cfg, to_opt):
        optmzr = None
        if optim_cfg.name == 'adam': 
            optmzr = optim.Adam(to_opt, lr=optim_cfg.lr)
        elif optim_cfg.name == 'sgd':
            optmzr = optim.SGD(to_opt, lr=optim_cfg.lr,
                               momentum=optim_cfg.momentum,
                               weight_decay=optim_cfg.weight_decay)
        elif optim_cfg.name == 'adamw':
            optmzr = optim.AdamW(to_opt, lr=optim_cfg.lr)
        else:
            raise ValueError("Unknown optimizer", optim_cfg.name)
        return optmzr

    @staticmethod
    def get_lr_sched(optim_cfg, optmzr, nepochs):
        if optim_cfg.lr_scheduler in ['cosine']:
            lr_sched = optim.lr_scheduler.CosineAnnealingLR
            lr_sched = lr_sched(optmzr, T_max=nepochs)
        elif optim_cfg.lr_scheduler in ['multistep']:
            fracs = optim_cfg.lr_milestone_fracs
            lr_milestones = [int(f * nepochs) for f in fracs]
            lr_gamma = optim_cfg.lr_gamma
            lr_sched = optim.lr_scheduler.MultiStepLR
            lr_sched = lr_sched(optmzr, lr_milestones, lr_gamma)
        elif optim_cfg.lr_scheduler is None:
            class lr_noop:
                def step(self): pass
            lr_sched = lr_noop()
        else:
            raise ValueError("Unknown lr-scheduler", optim_cfg.lr_scheduler)
        return lr_sched

    @staticmethod
    def get_loss_fn(lossfunc):
        if lossfunc == 'l2':
            return F.mse_loss
        elif lossfunc == 'bce':
            return F.binary_cross_entropy_with_logits
        elif lossfunc == 'xentropy':
            return F.cross_entropy

    @staticmethod
    def _epoch(payload, device, shard, trunk, data_schema, optimizer, tempr):
        """We don't use the teacher as of now but keeping the argument for fugure."""
        TR = _FineTuneTrainer
        tr_cfg = payload.train_cfg
        bz = tr_cfg.batch_size_gpu
        model = payload.module
        _kwargs = {'batch_size': bz , 'device': device}
        if hasattr(tr_cfg.transform, 'local_shuffle_buffer_size'):
            buffz = tr_cfg.transform.local_shuffle_buffer_size
            _kwargs['local_shuffle_buffer_size'] = buffz
        _iter = shard.iter_torch_batches(**_kwargs)
        x_key, y_key = data_schema['x_key'], data_schema['y_key']
        lossfn = TR.get_loss_fn(tr_cfg.lossfn)
        model.train()
        train_loss, total_batches = 0.0, 0.0
        for _, batch in enumerate(_iter):
            # get the input for the student model
            inx = batch[x_key].to(device, dtype=torch.float32)
            labels = batch[y_key].to(device)
            optimizer.zero_grad()
            outputs = model(inx)
            loss = lossfn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss
            total_batches = total_batches + 1
        meanfn = lambda x: x / total_batches
        means = [meanfn(x) for x in [train_loss]]
        ep_summary = {
            'loss': means[0].item(), 'ep_tempr': tempr
        }
        return ep_summary

    @staticmethod
    @torch.no_grad()
    def testmodel(rank, world_size, payload, dfctrl, split):
        """
        Compute the test accuracy w.r.t split specified. Always uses reference
        loaders.

        rank: Can be none in non-ddp mode
        # regular test acc : transform_test (testmodel + val + map)
        # noisy (IAI) acc : transform_noise_test (testmodel + val + noisy-map)
        # noisy (other) acc : transform_noise2_test (testmodel + other-map)
        # CF10-C acc: transform_c10c_test (testmodel + val-shift + c10c-map)
        """
        device = ray_get_device()
        jpl = payload
        try:
            bz = jpl.test_cfg.batch_size_gpu
        except AttributeError:
            bz = 256
        _ldrargs = {'split': split, 'rank': rank, 'device':device,
                    'ddp_world_size': world_size, 'transform_cfg': None}
        shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
        batch_iter = shard.iter_torch_batches(**{'batch_size':bz})
        schema = ray.get(dfctrl.get_data_schema.remote())
        x_key, y_key = schema['x_key'], schema['y_key']
        total_s, correct_s = 0, 0
        model = jpl.module
        model.eval(); model.to(device)
        for batchidx, batch in enumerate(batch_iter):
            x_batch, y_batch = batch[x_key], batch[y_key]
            logits = model(x_batch.to(device))
            if logits.shape[1] > 1:
                _, predicted = logits.max(1)
            else:
                predicted = torch.sigmoid(torch.squeeze(logits))
                predicted = (predicted > 0.5)
            iny = torch.squeeze(y_batch.to(device))#, non_blocking=True))
            correct = predicted.eq(iny)
            total_s += logits.size(0)
            correct_s += correct.sum().item()
        return (correct_s, total_s)

    @staticmethod
    def get_all_shifts_acc(rank, world_size, payload, dfctrl, split):
        TR = _FineTuneTrainer
        EASY_C_SHIFTS = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur']
        MEDIUM_C_SHIFTS = ['jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
        HARD_C_SHIFTS = ['gaussian_noise', 'glass_blur', 'impulse_noise', 'shot_noise']
        INET_C_SHIFTS = ['brightness', 'contrast', 'elastic_transform', 'fog', 
                        'frost', 'jpeg_compression', 'pixelate', 'snow']
        ALL_C_SHIFTS = EASY_C_SHIFTS + MEDIUM_C_SHIFTS + HARD_C_SHIFTS
        info = {}
        for shift in tqdm(ALL_C_SHIFTS):
            split_name = f'{split}-shift-{shift}'
            info[shift] = TR.testmodel(rank, world_size, payload, dfctrl, split_name)
        avg_acc = np.mean(list(info.values()))
        return info, avg_acc

    @staticmethod
    def finetune(rank, world_size, payload, dfctrl):
        # Setup before training
        TR = _FineTuneTrainer
        mlflow.set_experiment(experiment_name=payload.mlflow_expname)
        runid = payload.mlflow_runid
        device = ray_get_device()
        model, trunk = payload.module.to(device), payload.trunk.to(device)
        sd = TR.get_model_sd(payload)
        model.load_state_dict(sd)
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
        # Train loop
        start_time = time.time()
        for ep in range(train_cfg.num_epochs):
            tempr = TR.get_tempr(payload, ep)
            _eps = {'device': device, 'shard': shard, 'data_schema':
                    data_schema, 'payload': payload, 'trunk': trunk,
                    'optimizer': optmzr, 'tempr': tempr}
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
                tr_summary['noise_acc'] = corrn / totn
            tr_summary['epoch_duration'] = en_time - st_time
            if rank == 0:
                with mlflow.start_run(run_id=runid):
                    mlflow.log_metrics(tr_summary, step=ep)
        info = {'train_duration': time.time() - start_time}
        if (world_size > 1): model = model.module
        model = model.to('cpu')
        if rank == 0:
            with mlflow.start_run(run_id=runid):
                mlflow.log_metrics(info, step=train_cfg.num_epochs)
                name = 'ep-' + str(train_cfg.num_epochs)
                mlflow.pytorch.log_state_dict(model.state_dict().copy(), name)
        return 
