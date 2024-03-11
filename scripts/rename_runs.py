import mlflow
from mlflow.tracking.client import MlflowClient
from rich import print as pp

# expname='baseline-cifar10'
artifacts_root = '/data/dondennis/mlflow/mlartifacts/'
exp = mlflow.set_experiment(experiment_name=expname)
existing_runs = MlflowClient().search_runs(experiment_ids=[exp.experiment_id])
existing_runs = [r for r in existing_runs if r.info.run_name.startswith('migrated2')]

for run in existing_runs:
    params = dict(run.data.params)
    newname = run.info.run_name[len('migrated2-'):]
    artifact_uri = run.info.artifact_uri[len('mlflowartifacts:')+1:]
    artifacts_local = artifacts_root + artifact_uri
    pp({'old': run.info.run_name, 'new': newname})
    with mlflow.start_run(run_name=newname) as nrun:
        # Copy parameters, tags, metrics, etc.
        pp({'old': run.info.run_name, 'new': newname})
        tags = {k:v for k, v in run.data.tags.items() if not k.startswith('mlflow.')}
        mlflow.log_params(params)
        mlflow.log_metrics(run.data.metrics)
        mlflow.set_tags(tags)
        # assumign local storaeg
        mlflow.log_artifacts(artifacts_local)
