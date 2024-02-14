# Load the latest saved artifact for all runs and profile the model to get the flops.
import mlflow
import torch
import pandas as pd
from mlflow.tracking import MlflowClient

from ddistexps.utils import (
    load_mlflow_run_module, profile_module
)

SOURCE_EXP_NAMES = ['baseline/debug', 'distillation/debug', 
                    'baseline/cifar10', 'distillation/cifar10']

experiment_ids = []
for expname in SOURCE_EXP_NAMES:
    experiment_id = MlflowClient().get_experiment_by_name(expname).experiment_id
    experiment_ids.append(experiment_id)
    print(f"Found experiment {expname} with id {experiment_id}")

all_runs = MlflowClient().search_runs(experiment_ids=experiment_ids, max_results=10000)
# Go through each run and collect those that don't have flops.
runs_without_flops = []
for run in all_runs:
    if 'flops' in run.data.metrics:
        continue
    runs_without_flops.append(run)

print("Found", len(all_runs), "runs.")
print("Found", len(runs_without_flops), "runs without flops.")
for run in runs_without_flops:
    # Check if the run has the model artifact and remove 'param_str.txt' 
    artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
    artifacts = [a for a in artifacts if 'param_str.txt' not in a]
    if len(artifacts) == 0:
        print("Run", run.info.run_id, "has no model artifacts.")
        continue
    model = load_mlflow_run_module(run)
    inp_shape = eval(run.data.params['input_cfg.input_shape'])
    flops = profile_module('cuda', model, inp_shape)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({'flops': flops[0]})
