import mlflow
import torch
import pandas as pd
from mlflow.tracking import MlflowClient
from tqdm import tqdm

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
incorrect_out_runs = []
# For each run, check if the output shape is correct
for run in tqdm(all_runs):
    model, _ = load_mlflow_run_module(run)
    inp_shape = eval(run.data.params['input_cfg.input_shape'])
    inp = torch.randn((1,) + inp_shape)
    model = model.to('cuda')
    inp = inp.to('cuda')
    out = model(inp)
    out_class = out.shape[1]
    if out_class != 10:
        incorrect_out_runs.append(run)

print("Found", len(all_runs), "runs.")
print("Found", len(incorrect_out_runs), "runs with incorrect output shape.")
mlflow_client = MlflowClient()
for run in incorrect_out_runs:
    mlflow_client.delete_run(run.info.run_id)
