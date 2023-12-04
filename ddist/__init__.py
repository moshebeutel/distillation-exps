import ray
import mlflow

from pandas.api.types import is_numeric_dtype
from ddist.utils import CLog as lg
from ddist.utils import save_checkpoint

def get_driver_runid(expname):
    experiment = mlflow.set_experiment(expname)
    expid = experiment.experiment_id
    rname = 'ddist-driver'
    rid = mlflow.search_runs(experiment_ids=[expid],
                             filter_string=f"tags.mlflow.runName='{rname}'",
                             output_format='list')
    if len(rid) == 0:
        lg.info("No driver run found. Creating a new one.")
        # Start.run(None) -> Create new 
        return None, rname
    lg.info("Found existing driver run: (name:id)",rname, rid[0].info.run_id)
    return rid[0].info.run_id, rname

def ddist(expname, trunk, start_round, num_rounds, wlgen, out_dir, 
          max_recursion_depth=1):
    rid, rname = get_driver_runid(expname)
    with mlflow.start_run(run_id=rid, run_name=rname) as active_run:
        for round_id in range(start_round, num_rounds):
            lg.info(f"[red bold] Round {round_id}")
            ret = _round(round_id, trunk, wlgen, max_depth=max_recursion_depth)
            outpath = save_checkpoint(round_id, wlgen, out_dir)
            mlflow.log_artifact(outpath, artifact_path=f'round-{round_id}')
            if ret is None: break


def _round(round_id, trunk, wlgen, depth=0, max_depth=1):
    """
    """
    if depth == max_depth:
        msg= "[red]No candidate models found. "
        msg += f"Max-recursion depth hit {max_depth}."
        lg.info(msg)
        return None
    # Get candidate weak learners
    cref = wlgen.generate.remote(round_id)
    candidates = ray.get(cref)
    if candidates is None or len(candidates) == 0:
        lg.info(f"[red]No candidate models found." +
            f" (round: {round_id} depth: {depth+1}/{max_depth})")
        return None
    en = ray.get(wlgen.get_ensemble.remote())
    readyfut_list = wlgen.train.remote(candidates, trunk, en)
    _tr_cands = ray.get(readyfut_list)
    _ = [_freeze(mdl.module) for mdl in _tr_cands]
    assert len(_tr_cands) == len(candidates)
    candidates = _tr_cands
    lg.info("Picking a model for candidates . . ")
    _ref = wlgen.step.remote(round_id, candidates, trunk, en)
    wlpayload, wlrow, reasons = ray.get(_ref)
    if wlpayload is None:
        lg.info(f"[red]No weak learner found." + 
            f" (round: {round_id} depth: {depth}/{max_depth})")
        lg.info("Reasons:", reasons)
        lg.info("Generating more candidates from best current model")
        return _round(round_id, trunk, wlgen, depth=depth+1, max_depth=max_depth)
    lg.info(f"[green] Weak learner found (depth: {depth}/{max_depth})")
    _wlinfo = ray.get(wlgen.get_wl_profile.remote())
    lg.print_df(_wlinfo, title='WL-Info')
    return round_id


def _freeze(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    return model




