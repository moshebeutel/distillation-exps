from argparse import Namespace
import ray
import mlflow

import torch
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ray.air.util.torch_dist import TorchDistributedWorker
from ray.air.util.torch_dist import get_device as ray_get_device
from ray.data import ActorPoolStrategy

from ddist.utils import CLog as lg, namespace_to_dict
import numpy as np
from ddist.dispatch import BaseMapper
from ddistexps.utils import retrive_mlflow_run 


@ray.remote
class BaselineTrainer(BaseMapper):
    """
    Implements reduce ops on the mapper outputs.
    """

    def __init__(self, dfctrl, worker_cfg):

        super().__init__(_BaselineTrainer, worker_cfg.resource_req,
                         worker_cfg.world_size, worker_cfg.num_workers)
        self.dfctrl = dfctrl

    def train(self, payloads):
        lg.info(f"Multi-train started with {len(payloads)} candidate payloads")
        dfctrl = self.dfctrl
        fn_args = {'dfctrl': dfctrl}
        fn = _BaselineTrainer.train
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
        fn = _BaselineTrainer.testmodel
        map_results = self.map_workers(payloads, fn, fn_args)
        reduce_results_ = []
        for ref in map_results:
            # Reference to (num_correct, num_total) in that shard.
            vals = ray.get(ref)
            correct_samples = np.sum([elem[0] for elem in vals])
            total_samples = np.sum([elem[1] for elem in vals])
            acc = (correct_samples / total_samples) * 100.0
            reduce_results_.append(acc)
        return reduce_results_


@ray.remote
class _BaselineTrainer(TorchDistributedWorker):

    def __init__(self, name):
        self.basename = name
        self.name = name
        lg.info("Starting worker: ", name)

    def ready(self): return True

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
    def get_loss_fn(lossfunc):
        if lossfunc == 'l2':
            return F.mse_loss
        elif lossfunc == 'bce':
            return F.binary_cross_entropy_with_logits
        elif lossfunc == 'xentropy':
            return F.cross_entropy

    @staticmethod
    def _epoch(payload, device, shard, data_schema, optimizer, loss_fn):
        tr_cfg = payload.train_cfg
        bz = tr_cfg.batch_size_gpu
        module = payload.module
        buffz = None
        if hasattr(tr_cfg.transform, 'local_shuffle_buffer_size'):
            buffz = tr_cfg.transform.local_shuffle_buffer_size
        _kwargs = {'batch_size': bz, 'local_shuffle_buffer_size': buffz}
        _iter = shard.iter_torch_batches(**_kwargs)
        tot_loss, batches = 0.0, 0.0
        x_key, y_key = data_schema['x_key'], data_schema['y_key']
        module.train()
        for _, batch in enumerate(_iter):
            optimizer.zero_grad()
            inX = batch[x_key].to(device, dtype=torch.float32)
            iny = batch[y_key].to(device)
            predlogits = module(inX)
            loss = loss_fn(predlogits, iny)
            loss.backward()
            optimizer.step()
            tot_loss += loss
            batches += 1
        mean_epoch_loss = tot_loss / batches
        return mean_epoch_loss


    @staticmethod
    def train(rank, world_size, payload, dfctrl):
        '''
        Accelerated SGD based optimization. Train arguments are configured as
        part of the class-initialization in self.train-args
        '''
        _BL = _BaselineTrainer
        # We only start if this is a new run.
        mlflow.set_experiment(experiment_name=payload.mlflow_expname)
        runid = payload.mlflow_runid
        device = ray_get_device()
        model = payload.module.to(device)
        if (world_size > 1): 
            model = DDP(model)

        # Data Optimizer, lr-scheduler and criterion
        train_cfg = payload.train_cfg
        data_schema = ray.get(dfctrl.get_data_schema.remote())
        _ldrargs = {'split': 'train', 'rank': rank, 'ddp_world_size':
                    world_size, 'device': device,
                    'transform_cfg':train_cfg.transform}
        shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
        to_opt = [x for x in model.parameters() if x.requires_grad]
        optim_cfg = train_cfg.optim
        nepochs = train_cfg.num_epochs
        optmzr = _BL.get_optimizer(optim_cfg, to_opt)
        lr_sched = _BL.get_lr_sched(optim_cfg, optmzr, nepochs)
        lossfn = _BL.get_loss_fn(train_cfg.lossfn)
        tstfn = _BL.testmodel
        # Train loop
        tr_st_time = time.time()
        for ep in range(nepochs):
            _eps = {'device': device, 'shard': shard, 'data_schema':
                    data_schema, 'payload': payload, 'optimizer': optmzr,
                    'loss_fn':lossfn}
            start_time = time.time()
            mean_loss = _BaselineTrainer._epoch(**_eps)
            end_time = time.time()
            lr_sched.step()
            _info = {'epoch_loss': mean_loss, 'epoch_duration': end_time-start_time}
            if (ep % 10 == 0) or (ep == train_cfg.num_epochs - 1):
                val_acc = tstfn(rank, world_size, payload, dfctrl, 'val')
                _info['val_acc'] = val_acc
            if rank == 0:
                with mlflow.start_run(run_id=runid):
                    mlflow.log_metrics(_info, step=ep)
        # Log final train and val accuracy
        tr_acc = tstfn(rank, world_size, payload, dfctrl, 'reftrain')
        val_acc = tstfn(rank, world_size, payload, dfctrl, 'val')
        info = {'train_acc': tr_acc, 'val_acc': val_acc,
                'train_duration': time.time() - tr_st_time}
        if (world_size > 1): model = model.module
        model = model.to('cpu')
        if rank == 0:
            with mlflow.start_run(run_id=runid):
                mlflow.log_metrics(info, step=train_cfg.num_epochs)
                name = 'ep-' + str(train_cfg.num_epochs)
                X = torch.rand(1, *payload.input_cfg.input_shape)
                # mlflow.pytorch.log_state_dict(model.state_dict().copy(), name)
                # mlflow.pytorch.log_state_dict(model.state_dict().copy(), name)
                mlflow.pytorch.log_model(model, name, input_example=X.numpy())
        return 

    @staticmethod
    def testmodel(rank, world_size, payload, dfctrl, split):
        """
        Compute the test accuracy w.r.t the loader specified.
        model: If no model is provided, we return the ensemble-accuracy.
        """
        device = ray_get_device()
        jpl = payload 
        try:
            bz = jpl.test_cfg.batch_size_gpu
        except AttributeError:
            bz = 256

        _ldrargs = {'split': split, 'rank': rank, 'ddp_world_size': world_size,
                    'device': device}
        shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
        if split == 'train': # Because train is a pipeline not a ds.
            shard = next(shard.iter_epochs())
        batch_iter = shard.iter_torch_batches(**{'batch_size':bz})

        schema = ray.get(dfctrl.get_data_schema.remote())
        x_key, y_key = schema['x_key'], schema['y_key']
        total_s, correct_s = 0, 0
        model = jpl.module.eval().to(device)
        for batch_id, batch in enumerate(batch_iter):
            x_batch, y_batch = batch[x_key], batch[y_key]
            logits = model(x_batch.to(device))
            _, predicted = logits.max(1)
            iny = torch.squeeze(y_batch.to(device))#, non_blocking=True)
            correct = predicted.eq(iny)
            total_s += logits.size(0)
            correct_s += correct.sum().item()
        acc = 100.0 * correct_s / total_s
        return acc


