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
class DistilTrainer(BaseMapper):
    """
    Implements reduce ops on the mapper outputs.
    """
    def __init__(self, dfctrl, worker_cfg):
        self.dfctrl = dfctrl
        super().__init__(_DistilTrainer, worker_cfg.resource_req,
                         worker_cfg.world_size, worker_cfg.num_workers)

    def train(self, payloads):
        lg.info(f"Multi-train started with {len(payloads)} candidate payloads")
        dfctrl = self.dfctrl
        fn_args = {'dfctrl': dfctrl}
        fn = _DistilTrainer.train
        new_work = []
        for payload in payloads:
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
        reduce_results_ = []
        for vals in map_results:
            # Reference to (num_correct, num_total) in that shard.
            correct_samples = np.sum([elem[0] for elem in vals])
            total_samples = np.sum([elem[1] for elem in vals])
            acc = (correct_samples / total_samples) * 100.0
            reduce_results_.append(acc)
        return reduce_results_


@ray.remote
class _DistilTrainer(TorchDistributedWorker):
    def __init__(self, name):
        self.basename = name
        self.name = name
        lg.info("Starting trainer (worker):", name)

    @staticmethod
    def ready(): return True

    @staticmethod
    def get_optimizer(optim_cfg, to_opt):
        optmzr = None
        if optim_cfg.name not in ['adam','sgd']:
            raise ValueError("Unknown optimizer", optim_cfg.name)
        elif optim_cfg.name == 'adam': 
            optmzr = optim.Adam(to_opt, lr=optim_cfg.lr)
        elif optim_cfg.name == 'sgd':
            optmzr = optim.SGD(to_opt, lr=optim_cfg.lr, momentum=optim_cfg.momentum,
                       weight_decay=optim_cfg.weight_decay)
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
    def get_tempr(train_cfg, ep):
        nepochs = train_cfg.num_epochs
        temprcfg = train_cfg.loss_cfg.temperature
        i  = 0
        tempr_milestones = [int(v * nepochs) for v in temprcfg.milestone_fracs]
        while (i < len(tempr_milestones)) and (ep >= tempr_milestones[i]):
            i += 1
        return temprcfg.value * (temprcfg.gamma ** i)

    @staticmethod
    def distillation_loss(predlogits, tgtlogits, tempr):
        soft_tgt = F.softmax(tgtlogits / tempr, dim=-1)
        soft_pred = F.softmax(predlogits / tempr, dim=-1)
        approx_loss = -torch.sum(soft_tgt * soft_pred) / soft_pred.size()[0]
        approx_loss = approx_loss * (tempr**2)
        return approx_loss

    @staticmethod
    def _epoch(payload, device, shard, trunk, data_schema, optimizer, ep):
        _WL = _DistilTrainer
        tr_cfg = payload.train_cfg
        bz = tr_cfg.batch_size_gpu
        d_reg, x_reg = tr_cfg.loss_cfg.distil_reg, tr_cfg.loss_cfg.xentropy_reg
        module = payload.module
        _kwargs = {'batch_size': bz , 'device': device}
        if hasattr(tr_cfg.transform, 'local_shuffle_buffer_size'):
            buffz = tr_cfg.transform.local_shuffle_buffer_size
            _kwargs['local_shuffle_buffer_size'] = buffz
        _iter = shard.iter_torch_batches(**_kwargs)

        tot_loss, tot_d_loss, tot_x_loss, batches = 0.0, 0.0, 0.0, 0
        x_key, y_key = data_schema['x_key'], data_schema['y_key']
        module.train()
        tempr = _WL.get_tempr(payload.train_cfg, ep)
        for _, batch in enumerate(_iter):
            optimizer.zero_grad()
            inX = batch[x_key].to(device, dtype=torch.float32)
            iny = batch[y_key].to(device)
            acargs = {'device_type': 'cuda', 'dtype': torch.float16, 'enabled':
                      tr_cfg.use_amp}
            with torch.autocast(**acargs):
                predlogits = module(inX)
                tgtlogits = trunk(inX)
                x_loss = F.cross_entropy(predlogits, iny)
                d_loss = _WL.distillation_loss(predlogits, tgtlogits, tempr)
                loss = d_reg * d_loss + x_reg * x_loss
            loss.backward()
            optimizer.step()
            batches += 1
            tot_loss = tot_loss + loss
            tot_d_loss = tot_d_loss + d_loss
            tot_x_loss = tot_x_loss + x_loss
        meanfn = lambda x: x / batches
        means = [meanfn(x) for x in [tot_loss, tot_d_loss, tot_x_loss]]
        ep_summary = {
            'loss': means[0].item(), 'distil_loss': means[1].item(),
            'xentropy_loss': means[2].item(), 'ep_tempr': tempr
        }
        return ep_summary

    @staticmethod
    def train(rank, world_size, payload, dfctrl):
        # Setup before training
        WL = _DistilTrainer
        mlflow.set_experiment(experiment_name=payload.mlflow_expname)
        runid = payload.mlflow_runid
        device = ray_get_device()
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
        optmzr = WL.get_optimizer(optim_cfg, to_opt)
        lr_sched = WL.get_lr_sched(optim_cfg, optmzr, nepochs)
        # Train loop
        start_time = time.time()
        for ep in range(train_cfg.num_epochs):
            _eps = {'device': device, 'shard': shard, 'data_schema':
                    data_schema, 'payload': payload, 'trunk': trunk,
                    'optimizer': optmzr, 'ep': ep}
            st_time = time.time()
            tr_summary = WL._epoch(**_eps)
            lr_sched.step()
            en_time = time.time()
            if (ep % 10 == 0) or (ep == train_cfg.num_epochs - 1):
                payload.module = model
                corr, tot = WL.testmodel(rank, world_size, payload, dfctrl, 'val')
                val_acc = (corr / tot) * 100.0
                tr_summary['val_acc'] = val_acc
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

    @staticmethod
    @torch.no_grad()
    def testmodel(rank, world_size, payload, dfctrl, split):
        """
        Compute the test accuracy w.r.t split specified. Always uses reference
        loaders.
        model: If no model is provided, we return the ensemble-accuracy.

        rank: Can be none in non-ddp mode
        """
        device = ray_get_device()
        jpl = payload 
        try:
            bz = jpl.test_cfg.batch_size_gpu
        except AttributeError:
            bz = 256
        _ldrargs = {'split': split, 'rank': rank, 'device':device,
                    'ddp_world_size': world_size}
        shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
        batch_iter = shard.iter_torch_batches(**{'batch_size':bz})
        schema = ray.get(dfctrl.get_data_schema.remote())
        x_key, y_key = schema['x_key'], schema['y_key']
        total_s, correct_s = 0, 0
        model = jpl.module
        model.eval(); model.to(device)
        for batchidx, batch in enumerate(batch_iter):
            x_batch, y_batch = batch[x_key].to(device), batch[y_key].to(device)
            logits = model(x_batch)
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

