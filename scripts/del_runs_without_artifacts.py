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
runs_without_artifacts = []
for run in all_runs:
    # Check if the run has the model artifact and remove 'param_str.txt' 
    artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
    artifacts = [a for a in artifacts if 'param_str.txt' not in a]
    if len(artifacts) == 0:
        runs_without_artifacts.append(run)
        continue

print("Found", len(all_runs), "runs.")
print("Found", len(runs_without_artifacts), "runs without artifacts to be deleted.")
client = MlflowClient()
for run in runs_without_artifacts:
    client.delete_run(run.info.run_id)
print("Finished")
