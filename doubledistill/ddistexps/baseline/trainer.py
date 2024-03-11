import ray
import pandas as pd
import mlflow
from argparse import Namespace
from rich import print as rr

import torch
import time
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

from ray.air.util.torch_dist import TorchDistributedWorker
from ray.air.util.torch_dist import get_device as ray_get_device
# from ray.data import ActorPoolStrategy

from doubledistill.ddist.utils import (
    namespace_to_dict, flatten_dict
)
from doubledistill.ddist.dispatch import BaseMapper



@ray.remote
class BaselineReducer(BaseMapper):
    """
    Implements reduce ops on the mapper outputs.
    """
    def __init__(self, dfctrl, worker_cfg):

        super().__init__(BaselineWorker, worker_cfg.resource_req,
                         worker_cfg.world_size, worker_cfg.num_workers)
        self.dfctrl = dfctrl

    @staticmethod
    def _train_setup(run_cfg, dfctrl):
        """Converts run_cfg -> payload for training.
        Will always create a new run as continue is not implemented yet.
            payload.run_cfg
            payload.runid
            payload.model
        """
        dedup = run_cfg.meta.dedup_policy
        if dedup not in ['skip', 'recreate', 'continue']:
            raise ValueError("Unknown deduplication policy", dedup)
        input_shape = run_cfg.input_cfg.input_shape
        payload = Namespace(run_cfg=run_cfg)
        if dedup in ['skip', 'recreate']:
            # We only start if this is a new run.
            _kwargs = namespace_to_dict(run_cfg.module_cfg.kwargs)
            newmodel = run_cfg.module_cfg.fn(**_kwargs)
            payload.model = newmodel
        elif dedup == 'continue':
            # Get the appropriate checkpoint model.
            raise NotImplementedError("Continue not implemented yet")
        
        mlflow.set_experiment(experiment_name=run_cfg.meta.expname)
        with mlflow.start_run() as run:
            payload.runid = run.info.run_id
            mlflow.log_params(flatten_dict(namespace_to_dict(run_cfg)))
            X = torch.rand(1, *input_shape)
            mlflow.pytorch.log_model(payload.model, "model/initial", input_example=X.numpy())
        return payload
        
    @staticmethod
    def _train_reduce(results):
        runid = results[0].runid
        # Reduce the summary statistics.
        sums = [p.summary for p in results]
        summs_df = pd.DataFrame(sums)
        info = {}
        for col in summs_df.columns:
            # each column is touples. Sum the first and second elements.
            num = summs_df[col].apply(lambda x: x[0]).sum()
            denom = summs_df[col].apply(lambda x: x[1]).sum()
            info[col] = (num / denom) * 100.0
        with mlflow.start_run(run_id=runid):
            mlflow.log_metrics(info)
        return results

    def train(self, run_cfgs):
        rr(f"Multi-train started with {len(run_cfgs)} candidate payloads")
        dfctrl = self.dfctrl
        fn_args = {'dfctrl': dfctrl}
        fn = BaselineWorker.train
        setup_fn = BaselineReducer._train_setup
        setup_kwargs = {'dfctrl': dfctrl}
        teardown_fn = BaselineReducer._train_reduce
        map_results = self.map_workers(run_cfgs, fn, fn_args, setup_fn=setup_fn, 
                                       setup_kwargs=setup_kwargs, teardown_fn=teardown_fn)
        map_results_ = [res[0] for res in map_results]
        rr("Multi-train: Finished")
        return map_results_


@ray.remote
class BaselineWorker(TorchDistributedWorker):
    def __init__(self, name):
        self.name = name

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
    def _epoch(rank, model, run_cfg, device, shard, data_schema, optimizer, loss_fn):
        tr_cfg = run_cfg.train_cfg
        bz = tr_cfg.batch_size_gpu
        buffz = None
        if hasattr(tr_cfg.transform, 'local_shuffle_buffer_size'):
            buffz = tr_cfg.transform.local_shuffle_buffer_size
        _kwargs = {'batch_size': bz, 'local_shuffle_buffer_size': buffz}
        _iter = shard.iter_torch_batches(**_kwargs)
        tot_loss, batches = 0.0, 0.0
        x_key, y_key = data_schema['x_key'], data_schema['y_key']
        model.train()
        for _, batch in enumerate(_iter):
            optimizer.zero_grad()
            inX = batch[x_key].to(device, dtype=torch.float32)
            iny = batch[y_key].to(device)
            predlogits = model(inX)
            loss = loss_fn(predlogits, iny)
            loss.backward()
            optimizer.step()
            tot_loss += loss
            batches += 1
        batches = 1
        mean_epoch_loss = tot_loss / batches
        return mean_epoch_loss

    @staticmethod
    def train(rank, world_size, payload, dfctrl):
        '''
        Train payload using dataflow. Expects:
            payload.run_cfg, payload.model, payload.runid
        '''
        _BL = BaselineWorker
        device = ray_get_device()
        run_cfg, model = payload.run_cfg, payload.model.to(device)
        runid = payload.runid
        # Setup DDP for model for multi-gpu training
        if (world_size > 1): model = DDP(model)
        # Data shard for this rank.
        train_cfg = run_cfg.train_cfg
        data_schema = ray.get(dfctrl.get_data_schema.remote())
        _ldrargs = {'split': 'train', 'rank': rank, 'ddp_world_size':
                    world_size, 'device': device,
                    'transform_cfg':train_cfg.transform}
        shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
        # Optimizer
        to_opt = [x for x in model.parameters() if x.requires_grad]
        optim_cfg = train_cfg.optim
        nepochs = train_cfg.num_epochs
        optmzr = _BL.get_optimizer(optim_cfg, to_opt)
        lr_sched = _BL.get_lr_sched(optim_cfg, optmzr, nepochs)
        lossfn = _BL.get_loss_fn(train_cfg.lossfn)
        tstfn = _BL.testmodel
        # Train loop
        best_val_acc = 0.0
        train_st_time = time.time()
        for ep in range(nepochs):
            # Epoch
            _eps = {'rank': rank, 'device': device, 'shard': shard, 'data_schema':
                    data_schema, 'model': model, 'run_cfg': run_cfg,
                    'optimizer': optmzr, 'loss_fn':lossfn}
            start_time = time.time()
            mean_loss = BaselineWorker._epoch(**_eps)
            end_time = time.time()
            lr_sched.step()
            _info = {f'epoch_loss{rank}': mean_loss,
                     f'epoch_duration{rank}': end_time-start_time}
            with mlflow.start_run(run_id=runid):
                mlflow.log_metrics(_info, step=ep)
            # Continue if not a validation epoch
            is_validation_ep = (ep % 10 == 0) or (ep == train_cfg.num_epochs - 1)
            if not is_validation_ep:
                continue
            
            # Validation and Model Logging
            ckpt_model = False
            count, tot = tstfn(rank, world_size, payload, dfctrl, 'val')
            val_acc = (count / tot) * 100.0
            if val_acc > best_val_acc:
                ckpt_model = True
            best_val_acc = max(best_val_acc, val_acc)
            val_info = {f'val_acc{rank}': val_acc, f'val_acc{rank}_best': best_val_acc}
            with mlflow.start_run(run_id=runid):
                mlflow.log_metrics(val_info, step=ep)
            if ckpt_model is True:
                if world_size > 1:
                    sd = model.module.state_dict().copy()
                else:
                    sd = model.state_dict().copy()
                with mlflow.start_run(run_id=runid):
                    mlflow.pytorch.log_state_dict(sd, f'state_dict/best{rank}')
        train_end_time = time.time() - train_st_time
        if world_size > 1:
            model = model.module
        # Log final model
        with mlflow.start_run(run_id=runid):
            final_sd = model.state_dict().copy()
            mlflow.log_metrics({f'train_duration{rank}': train_end_time})
            mlflow.pytorch.log_state_dict(final_sd, f'state_dict/final{rank}')
        # Final val_acc and best val_acc
        final_val = tstfn(rank, world_size, payload, dfctrl, 'val')
        artifact_uri = f'runs:/{runid}/state_dict/best{rank}'
        bestsd = mlflow.pytorch.load_state_dict(artifact_uri)
        model.load_state_dict(bestsd)
        payload.model = model
        best_val = tstfn(rank, world_size, payload, dfctrl, 'val')
        info = {'val_acc_final': final_val, 'val_acc_best': best_val}
        # Return summary to be reduced
        model.load_state_dict(final_sd)
        payload.model = model.to('cpu')
        payload.summary = info
        return payload

    @staticmethod
    def testmodel(rank, world_size, payload, dfctrl, split):
        """
        Compute the correct and total samples w.r.t the specified shard.
        model: If no model is provided, we return the ensemble-accuracy.
        """
        device = ray_get_device()
        jpl = payload.run_cfg
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
        model = payload.model.to(device)
        for batch_id, batch in enumerate(batch_iter):
            x_batch, y_batch = batch[x_key], batch[y_key]
            logits = model(x_batch.to(device))
            _, predicted = logits.max(1)
            iny = torch.squeeze(y_batch.to(device))#, non_blocking=True)
            correct = predicted.eq(iny)
            total_s += logits.size(0)
            correct_s += correct.sum().item()
        return correct_s, total_s


