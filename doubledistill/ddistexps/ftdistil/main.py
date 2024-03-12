# 
# CLIP based fine tuning on pretrained models. We 
# wish to understand the effect of CLIP supervised fine-tuning on
# out of distribution performance at various FLOPS.
#
# 1. Programatically pick some models to fine-tune
# 2. Fine-tune the models using CLIP supervised fine-tuning
# 3. Evaluate the models on various OOD datasets

import ray
import os
import time
# import mlflow
import numpy as np
from argparse import ArgumentParser
from rich import print

from doubledistill.ddist.data import get_dataset
from doubledistill.ddist.utils import (
    spec_to_prodspace, dict_to_namespace
)

from doubledistill.ddistexps.utils import (
    get_dataflow
)

from doubledistill.ddistexps.ftdistil.trainer import FineTuneTrainer
from doubledistill.ddistexps.ftdistil.expcfg import EXPERIMENTS


def get_pretrained_runs(src_exp_names, nclusters=8, nmodels_per_cluster=2):
    """Cluster the models based on FLOPS and validation accuracy.
    Pick the top nmodels_per_cluster from each cluster.

    Returns list of picked runs.
    """
    SOURCE_EXPS = src_exp_names
    # Get all the runs from these experiments. We assume `flops` have already
    # been populated for these.
    all_exp_ids = []
    for expname in SOURCE_EXPS:
        print("Getting experiment:", expname)
        # eid = mlflow.get_experiment_by_name(expname).experiment_id
        # all_exp_ids.append(eid)

    # all_runs_df = mlflow.search_runs(experiment_ids=all_exp_ids, max_results=10000,
    #                                  filter_string="attributes.status = 'FINISHED'")
    data1 = all_runs_df[['run_id', 'metrics.flops', 'metrics.val_acc', 'metrics.train_duration']]
    # Remove rows with NaN
    data1 = data1.dropna()
    data1.columns = ['runid', 'flops', 'val_acc', 'train_duration']
    from sklearn.cluster import KMeans
    kmean = KMeans(n_clusters=nclusters)
    kmean.fit(data1[['flops', 'val_acc']])
    # Set cluster ids so that they are in order of flops
    centers = kmean.cluster_centers_
    centers = centers[np.argsort(centers[:, 0])]
    kmean.cluster_centers_ = centers
    data1['cluster'] = kmean.predict(data1[['flops', 'val_acc']])

    fn = lambda x: x.nlargest(nmodels_per_cluster, 'val_acc')
    cluster_models = data1.groupby('cluster').apply(fn)
    cluster_models = cluster_models.reset_index(drop=True)
    retvals = cluster_models['runid'].values
    return list(retvals)


if __name__ == '__main__':
    parser = ArgumentParser()
    msg = "The exeriment name as expcfg.EXPERIMENTS."
    parser.add_argument("--expname", type=str, help=msg, required=True,
                        default=None)

    expname = parser.parse_args().expname
    spec = EXPERIMENTS[expname]
    # Attach sub-directory to expname
    expname = 'ftdistil/' + expname
    spec['mlflow_expname'] = [expname]
    meta = spec['meta']
    src_run_grid = get_pretrained_runs(meta['src_exp_names'][0]) 
    spec['src_run'] = src_run_grid
    
    dataset = spec['dataflow']['data_set']
    ds_meta = get_dataset(dataset).metadata

    prod_space = spec_to_prodspace(**spec)
    payloads = [dict_to_namespace(p) for p in prod_space]
    meta = dict_to_namespace(meta)

    dfnamespace = 'FDistilDataFlow'
    ray.init(namespace=dfnamespace)
    tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    # mlflow.set_tracking_uri(tracking_uri)
    # print("[green bold] Connecting to mlflow using:", tracking_uri)

    dflow = prod_space[0]['dataflow']
    ref = get_dataflow.remote(dflow, meta.worker_cfg.world_size,
                              engine='augmented')
    dfctrl = ray.get(ref)
    print("DF Actor ready:", ray.get(dfctrl.ready.remote()))
    dispatch_kwargs = { 'dfctrl': dfctrl, 'worker_cfg': meta.worker_cfg}

    dispatch = FineTuneTrainer.remote(**dispatch_kwargs)
    print("Starting Finetune-Dispatch")
    wlref = dispatch.finetune.remote(payloads)
    st_time = time.time()
    ray.get(wlref)
    info_ = {'experiment': expname, 'num_payloads': len(payloads),
            'total-duration': time.time() - st_time}
    print("Experiment completed:", info_)

