# Easiest way to fix runs is to create a new run and copy/fix parameters;
import mlflow
from mlflow.tracking.client import MlflowClient
from doubledistill.ddistexps.utils import param_hash
from rich import print as pp

expname='debug'
artifacts_root = '/data/dondennis/mlflow/mlartifacts/'
exp = mlflow.set_experiment(experiment_name=expname)
existing_runs = MlflowClient().search_runs(experiment_ids=[exp.experiment_id])
existing_runs = [r for r in existing_runs if not r.info.run_name.startswith('migrated')]

for run in existing_runs:
    params = dict(run.data.params)
    metadict = {k:v for k,v in params.items() if k.startswith('meta.')}
    migrate_keys = ['meta.data_set', 'meta.preprocessor', 'meta.read_parallelism']
    for k in metadict.keys():
        if k in migrate_keys:
            params['dataflow.' + k[len('meta.'):]] = params[k]
        del params[k]
    del params['param_hash']
    new_str, new_hash = param_hash(params)
    params['param_hash'] = new_hash
    params = params | {k: v for k,v in metadict.items() if k not in migrate_keys}
    # Start a new run with this name and copy everything else over.
    newname = 'migrated-' + run.info.run_name
    artifact_uri = run.info.artifact_uri[len('mlflowartifacts:')+1:]
    artifacts_local = artifacts_root + artifact_uri
    with mlflow.start_run(run_name=newname) as nrun:
        # Copy parameters, tags, metrics, etc.
        pp({'old': run.info.run_name, 'new': newname})
        tags = {k:v for k, v in run.data.tags.items() if not k.startswith('mlflow.')}
        mlflow.log_params(params)
        mlflow.log_metrics(run.data.metrics)
        mlflow.set_tags(tags)
        # assumign local storaeg
        mlflow.log_artifacts(artifacts_local)
        mlflow.log_text(new_str, 'param_str-migrated.txt')
