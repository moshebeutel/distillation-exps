import mlflow
from mlflow.tracking.client import MlflowClient
from rich import print as pp

expname='baseline-cifar10'
# expname = 'debug'
artifacts_root = '/data/dondennis/mlflow/mlartifacts/'


exp = mlflow.set_experiment(experiment_name=expname)
existing_runs = MlflowClient().search_runs(experiment_ids=[exp.experiment_id])
to_delete_runs = [r for r in existing_runs if r.info.run_name.startswith('migrated2')]
client = MlflowClient()
run_names = [r.info.run_name for r in to_delete_runs]
pp("Will delte the folliwng runs:")
pp(run_names)
for run in to_delete_runs:
    client.delete_run(run.info.run_id)
pp("Finished")
