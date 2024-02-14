import pandas as pd
import numpy as np
import torch

from ddistexps.utils import load_mlflow_run_module, profile_module
from ddistexps.utils import param_hash
from ddist.utils import spec_to_prodspace, dict_to_namespace
from ddist.models import Ensemble
from mlflow.tracking.client import MlflowClient


def profile_runs(runlist, input_shape):
    """Profiles a list of runs and returns a dataframe with the metrics."""
    metrics = [r.data.metrics for r in runlist]
    metricsdf = pd.DataFrame(metrics)
    metricsdf['runid'] = [r.info.run_id for r in runlist]
    for run in runlist:
        # Fetch the artifact directory and the model from it
        model = load_mlflow_run_module(run)
        flops = profile_module('cuda', model, input_shape)
        metricsdf.loc[metricsdf['runid'] == run.info.run_id, 'flops'] = flops[0]
    return metricsdf


def fetch_valid_runs(expnames, acc_threshold=20.0):
    """Fetches all runs from the given experiment names and
    returns a dataframe with the metrics of the runs that 
    have a validation accuracy higher than the given threshold
    and have valid artifacts to load.
    """
    experiment_ids = []
    for expname in expnames:
        experiment_id = MlflowClient().get_experiment_by_name(expname).experiment_id
        experiment_ids.append(experiment_id)
        print(f"Found experiment {expname} with id {experiment_id}")

    existing_runs = MlflowClient().search_runs(experiment_ids=experiment_ids, max_results=10000)
    print(f"Found {len(existing_runs)} runs")
    # Filter out runs without artifacts
    valid_runs = []
    for run in existing_runs:
        # Check if the run has the model artifact and remove 'param_str.txt' 
        artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
        artifacts = [a for a in artifacts if 'param_str.txt' not in a]
        if len(artifacts) == 0:
            continue
        valid_runs.append(run)
    print(f"Found {len(valid_runs)} runs with valid artifacts")
    metrics = [r.data.metrics for r in valid_runs]
    metricsdf = pd.DataFrame(metrics)
    metricsdf['runid'] = [r.info.run_id for r in valid_runs]
    valid_runs = metricsdf[metricsdf['val_acc'] > acc_threshold]
    print(f"Found {len(valid_runs)} valid runs acc-threshold: {acc_threshold}")
    validrunids = valid_runs['runid']
    validruns = [MlflowClient().get_run(runid) for runid in validrunids]
    return validruns

def testmodel(dfctrl, module, device, bz=256):
    _ldrargs = {'split': 'val', 'rank': 0, 'device': device, 'ddp_world_size': 1}
    shard = ray.get(dfctrl.getshard.remote(**_ldrargs))
    batch_iter = shard.iter_torch_batches(**{'batch_size':bz})
    schema = ray.get(dfctrl.get_data_schema.remote())
    x_key, y_key = schema['x_key'], schema['y_key']
    total_s, correct_s = 0, 0
    model = module
    model.eval(); model.to(device)
    for batchidx, batch in enumerate(batch_iter):
        x_batch, y_batch = batch[x_key].to(device), batch[y_key].to(device)
        logits = model(x_batch)
        if logits.shape[1] > 1:
            _, predicted = logits.max(1)
        else:
            predicted = torch.sigmoid(torch.squeeze(logits))
            predicted = (predicted > 0.5)
        iny = torch.squeeze(y_batch.to(device))
        correct = predicted.eq(iny)
        total_s += logits.size(0)
        correct_s += correct.sum().item()
    return (correct_s, total_s)



def get_en_stats(dfctrl, en_runid_dict):
    """ensemble specified as {round: runid}"""
    # Fetch each run, anddef profile_runs(runlist):
    en_run_dict = {}
    en_module_dict = {}
    for k, runid in en_runid_dict.items():
        run = MlflowClient().get_run(runid)
        module = load_mlflow_run_module(run)
        en_run_dict[k] = run
        en_module_dict[k] = module
    
    # compute flops for only the trainable paramters. Ensemble
    # flops can be computed as sum.
    en_trainable_flops = []
    en_val_acc = []
    # Access rounds in order
    n_rounds = len(en_runid_dict.keys())
    ensemble_module_list = []
    for round in range(n_rounds):
        # Fetch the artifact directory and the model from it
        module = en_module_dict[str(round)]
        if round == 0:
            trainable_module = module
        else:
            trainable_module = module.trainable
        
        flops = profile_module('cuda', trainable_module, (3, 32, 32))
        en_trainable_flops.append(flops[0])
        ensemble_module_list.append(module)
        en = Ensemble(ensemble_module_list)
        corr, tott = testmodel(dfctrl, en, 'cuda')
        en_val_acc.append((corr/tott) * 100.0)
    en_flops = []
    for i in range(1, n_rounds+1):
        en_flops.append(sum(en_trainable_flops[:i]))
    
    en_stats = pd.DataFrame({'round': list(en_runid_dict.keys()),
                                'en_val_acc': en_val_acc,
                                'en_flops': en_flops})
    return en_stats