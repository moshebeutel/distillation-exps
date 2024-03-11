# Load the trainable module from a composed model and profile it to get the flops.
# Adds it as trainable-flops key
import mlflow
import torch
from rich import print
from mlflow.tracking import MlflowClient

from doubledistill.ddist.models import Composer, Lambda

from doubledistill.ddistexps.utils import (
    load_mlflow_run_module, profile_module
)

SOURCE_EXP_NAMES = [
    'composition/debug',
]

experiment_ids = []
for expname in SOURCE_EXP_NAMES:
    experiment_id = MlflowClient().get_experiment_by_name(expname).experiment_id
    experiment_ids.append(experiment_id)
    print(f"Found experiment {expname} with id {experiment_id}")

all_runs = MlflowClient().search_runs(experiment_ids=experiment_ids, max_results=10000)
# Go through each run and collect those that don't have flops.
runs_without_trainable_flops = []
for run in all_runs:
    if 'flops_trainable' in run.data.metrics:
        continue
    runs_without_trainable_flops.append(run)

print("Found", len(all_runs), "runs.")
print("Found", len(runs_without_trainable_flops), "runs without trainable flops.")
all_info = []
for run in runs_without_trainable_flops:
    # Check if the run has the model artifact and remove 'param_str.txt' 
    artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id)]
    artifacts = [a for a in artifacts if 'param_str.txt' not in a]
    if len(artifacts) == 0:
        print("Run", run.info.run_id, "has no model artifacts.")
        continue
    model = load_mlflow_run_module(run)
    inp_shape = eval(run.data.params['input_cfg.input_shape'])
    # Make sure the model is a composed model
    if type(model) != Composer:
        print("Run", run.info.run_id, "has no composed model.")
        continue
    
    connection = run.data.params['compose_cfg.conn_name']
    trainable = model.trainable
    all_flops = profile_module('cuda', model, inp_shape)
    trainable_flops = profile_module('cuda', trainable, inp_shape)
    info = {"run_id": run.info.run_id, 'connection': connection, "flops_all": all_flops[0],
           "flops_trainable": trainable_flops[0]}
    
    # Get flops of pre-block, layer-block, post-block
    inp = torch.randn((1,) + inp_shape).to('cuda')
    pre_block = trainable.pre_block
    try:
        # Pre block could be a functional
        pre_block = pre_block.to('cuda')
    except AttributeError:
        pre_block = Lambda(pre_block)
    flops = profile_module('cuda', pre_block, inp.shape, batched_inp_shape=True)
    info['flops_pre'] = flops[0]

    pre_out = pre_block(inp)
    layer_block = trainable.layer_block
    flops = profile_module('cuda', layer_block, pre_out.shape, batched_inp_shape=True)
    info['flops_layer'] = flops[0]

    layer_out = layer_block(pre_out)
    post_block = trainable.post_block
    try:
        # Pre block could be a functional
        post_block = post_block.to('cuda')
    except AttributeError:
        post_block = Lambda(post_block)
    flops = profile_module('cuda', post_block, layer_out.shape, batched_inp_shape=True)
    info['flops_post'] = flops[0]

    all_info.append(info)
    # log the flops to the run
    log_keys = ['flops_all', 'flops_trainable', 'flops_pre', 'flops_layer', 'flops_post']
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_metrics({k: info[k] for k in log_keys})

# df = pd.DataFrame(all_info)
# for key in log_keys:
#     df[f'p-{key}'] = df[key] / df['flops_all'] * 100.0

# columns = ['run_id', 'connection']
# columns += [f'p-{key}' for key in log_keys]
# print(df[columns])
