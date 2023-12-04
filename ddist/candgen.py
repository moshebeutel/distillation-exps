import numpy as np
import pandas as pd
import torch
import os
import mlflow

import ray
import torch.nn as nn
from sklearn.linear_model import LinearRegression

from ddist.utils import CLog as lg
from ddist.utils import flatten_dict
from ddist.utils import spec_to_prodspace, dict_to_namespace, namespace_to_dict
from ddist.models import Ensemble, ConnectionContainer as CC

def fwd_noconnection(self, x):
    "No reuse, just the latest model. Latest model is stored in trainable"
    return self.trainable(x)

class CandGenBase:
    class State:
        def __init__(self):
            self.df_prof = None
            self.df_measured = None
            self.df_predicted = None

            self.metric_ub_list = []
            self.metric_column = None
            self.pid = 0

    def state_dict(self):
        return self.state.__dict__

    def load_state_dict(self, sd):
        if self.__use_new_db is True:
            del sd['df_prof']
        self.state.__dict__.update(sd)


    def __init__(self, file_prof, dispatch, expname, wlsearch_cfg, 
                 metric_ub=None, metric_column='total_duration', idcol='module-id',
                 update_db=True, ignore_mixing=False):
        df_prof = pd.read_pickle(file_prof)
        self.__use_new_db = update_db

        expected_columns = [idcol, metric_column, 'module_class',
                            'module_kwargs']
        for col in expected_columns:
            if col in df_prof.columns: 
                continue
            msg = f"Expected column {col} not found in" 
            msg += f"db.columns: {df_prof.columns}"
            raise ValueError(msg)

        idvals = df_prof[idcol].values
        uids = np.unique(idvals)
        assert len(uids) == len(idvals), f"Duplicate module-ids {len(uids)}|{len(idvals)}"
        df_prof.set_index(idcol, inplace=True)
        # Create empty dataframe for trained.

        self.cols_measured = [idcol, 'round', 'val_acc', 'en_val_acc',
                              'uw_en_val_acc', 'eps_t',
                              'norm_dirtn', 'mixing_weight', 'gamma_t','is_wl',
                             'payload_obj', 'connection', metric_column]
        self.cols_predicted = [idcol, 'round', 'connection', 'predicted_en_acc']
        df_measured = pd.DataFrame(columns=self.cols_measured)
        df_predicted = pd.DataFrame(columns=self.cols_predicted)
        if metric_ub is None:
            metric_ub = 2 * np.min(df_prof[metric_column].values)
        self.state = CandGenBase.State()
        self.state.df_prof = df_prof
        self.state.df_measured = df_measured
        self.state.df_predicted = df_predicted
        self.state.metric_ub_list.append(metric_ub)
        self.state.metric_column = metric_column
        self.connections_list = wlsearch_cfg['connections']

        self.expname = expname
        self.search_cfg = wlsearch_cfg
        self.ncands_to_generate = wlsearch_cfg['ncands_to_generate']
        self.dispatch = dispatch
        self.idcol = idcol
        self.ignore_mixing = ignore_mixing

    def get_db(self):
        return self.state.df_prof

    def __modulecfgs_x_searchcfgs(self, module_cfgs, verbose_depth=1):
        """Module_cfgs: Dict of non-closured config.
        spec: A dict with keys that end with _cfg and values that are lists.

        Returns a configuration dict with all possible combinations of
        module_cfg and provided `_cfg` in spec"""
        spec = self.search_cfg
        module_cfgs = module_cfgs.copy()
        if type(module_cfgs) is not dict:
            # List of dicts
            module_cfgs = module_cfgs.to_dict('records')

        cfg_keys = [x for x in spec.keys() if x.endswith('_cfg')]
        _spec = {c:spec[c] for c in cfg_keys}
        _spec = _spec | {'module_cfg': module_cfgs}
        ret = spec_to_prodspace(verbose_depth=verbose_depth,
                                       **_spec) 
        return ret

    def default_base_learners(self):
        df = self.state.df_prof
        metric_column = self.state.metric_column
        metric_ub = self.state.metric_ub_list[-1]
        cands = df[df[metric_column] <= metric_ub].copy()
        cands[self.idcol] = cands.index.values
        return cands

    def generate(self, round_id):
        """
        If no measurements have been made, then we just retun the best model in
        terms of metric_ub. (round 0)
        Returns None if no more candidates exist.
        """
        num_candidates = self.ncands_to_generate
        metric_column = self.state.metric_column
        metric_vals = self.state.df_prof[metric_column].values
        max_metric_ub = np.max(metric_vals)
        new_pred = self.fit_and_predict(round_id)
        if round_id == 0 and new_pred is None:
            cands = self.default_base_learners()
            cands = self.__modulecfgs_x_searchcfgs(cands)
            return self.__make_candidates(cands[:num_candidates])
        elif new_pred is None:
            return None
        curr_ub = self.state.metric_ub_list[-1]
        quant = np.mean((metric_vals < curr_ub))
        while True:
            new_pred_now = new_pred[new_pred[metric_column] <= curr_ub]
            if len(new_pred_now) >= 2 * num_candidates:
                break
            _scale = 2.0
            if quant >= 1.0: break
            quant = np.min([quant * _scale, 1.0])
            newub = np.quantile(metric_vals, quant)
            lg.info(f"Relaxing metric ub: {curr_ub}->{newub} (quantile:{quant})")
            curr_ub = newub
        new_pred = new_pred_now
        if curr_ub > max_metric_ub: lg.info("We are considering all candidates!")
        if curr_ub != self.state.metric_ub_list[-1]:
            self.state.metric_ub_list.append(curr_ub)
            lg.info("Updated metric ub list:", self.state.metric_ub_list)
        df_predicted = self.state.df_predicted
        if len(df_predicted) == 0:
            mdf = new_pred
        else:
            mdf = pd.concat((df_predicted, new_pred), axis=0)
        self.state.df_predicted = mdf 
        cands = self.__modulecfgs_x_searchcfgs(new_pred)
        new_pred_topk = self.pick_topk_candidates(round_id, cands, num_candidates)
        if new_pred_topk is None:
            return None
        return self.__make_candidates(new_pred_topk)

    def refresh_measurements(self):
        """Refreshes the measured dataframe with the latest measurements: 
        Uses the saved payload to get the new ensemble and validation
        accuracies"""
        df = self.state.df_measured.copy()
        self.state.df_measured = pd.DataFrame(columns=self.cols_measured)
        rounds = df['round'].values
        self.refresh_residuals()
        for round in np.unique(np.sort(rounds)):
            lg.info(f"[green] Updating round: {round}")
            payloads = df[df['round'] == round]['payload_obj'].values
            if len(payloads) == 0:
                lg.info("No payloads found for round:", round)
                return False
            ret = self.step(round, payloads)
            wlpayload, wlrow, reasons = ret
            if wlpayload is None:
                lg.info("No weak learner found in round:", round)
                return False
            lg.info("Found weak learner in round:", round)
        return True

    def __get_exclude_mask(self, newpred_flat, round_id):
        flat_search_cfg = spec_to_prodspace(verbose_depth=0, **self.search_cfg)
        flat_search_keys = []
        for elem in flat_search_cfg:
            keys = flatten_dict(elem).keys()
            for k in keys:
                if k in flat_search_keys:
                    continue
                flat_search_keys.append(k)
        flat_search_keys = [x for x in flat_search_keys if '_cfg'in x]
        flat_search_keys.append('module_cfg.connection')
        df_mr = self.state.df_measured
        df_mr = df_mr[df_mr['round'] == round_id]
        already_measured_base_id = df_mr[self.idcol].values
        already_measured_base_id = np.unique(already_measured_base_id)
        def __exclude(row):
            idval = row['module_cfg.' + self.idcol]
            if idval not in already_measured_base_id: return False
            row_measured = df_mr[df_mr[self.idcol] == idval].copy()
            flat_measured = [d['payload_obj'] for d in row_measured.to_dict('records')]
            flat_measured = [namespace_to_dict(d) for d in flat_measured]
            flat_measured = [flatten_dict(d) for d in flat_measured]
            diff_keys = []
            for mdict in flat_measured:
                equal_to_row = True
                for k in flat_search_keys:
                    newk = (k not in mdict.keys()) or (k not in row.keys())
                    if newk or row[k] != mdict[k]:
                        equal_to_row = False
                        diff_keys.append(k)
                        info = {'key': k, 'is_missing_key': newk}
                        if not newk:
                            info['row_val'] = row[k]
                            info['measured_val'] = mdict[k]
                        lg.info("Diff key:", info)
                        break
                if equal_to_row is True:
                    lg.info("Found duplicate:")
                    return True
            return False

        mask_exclude = newpred_flat.apply(lambda x: __exclude(x), axis=1) 
        return mask_exclude

    def pick_topk_candidates(self, round_id, new_pred, num_candidates):
        """Randomized top-k picking based on prediction accuracy"""
        metric_key = 'module_cfg.' + self.state.metric_column
        # metric_key = 'module_cfg.' + self.state.metric_column
        # id_key = 'module_cfg.' + self.idcol
        df_mr = self.state.df_measured
        df_mr = df_mr[df_mr['round'] == round_id]
        new_pred_flat = [flatten_dict(d) for d in new_pred.copy()]
        new_pred_flat = pd.DataFrame(new_pred_flat)
        _exclude_mask = self.__get_exclude_mask(new_pred_flat, round_id)
        while True:
            curr_ub = self.state.metric_ub_list[-1]
            include_mask = new_pred_flat[metric_key] <= curr_ub
            include_mask = np.logical_and(include_mask, np.logical_not(_exclude_mask))
            exclude_mask = np.logical_not(include_mask)
            if np.sum(include_mask) >= num_candidates:
                break
            _scale = 2.0
            lg.info(f"Curr_ub: {curr_ub:.5f} -> {curr_ub * _scale:.5f}")
            curr_ub = curr_ub * _scale
            self.state.metric_ub_list.append(curr_ub)
            if curr_ub > _scale * np.max(new_pred_flat[metric_key].values):
                lg.info("UB relaxed to max-value")
                break
        lg.info("Configurations to measure:", np.sum(include_mask))
        if np.sum(include_mask) == 0:
            lg.info("No more configurations to test")
            return None
        curr_ub = self.state.metric_ub_list[-1]
        p = new_pred_flat['module_cfg.predicted-en-acc'].values
        p[exclude_mask] = 0.0
        # We are out of models 
        _random_frac = 0.2
        cand_p = int(num_candidates * (1 - _random_frac))
        try:
            # Get distribution based on predicted-en-acc
            p = p + np.abs(np.min(p))
            p = p / np.sum(p)
            idx_1 = np.random.choice(len(p), size=cand_p, replace=False, p=p)
        except Exception as e:
            lg.warning("Could not sample from predicted distribution. Will use"
                       + "uniform distribution")
            lg.warning(e)
            idx_1 = []
        remaining_ids = np.array([x for x in range(len(p)) if x not in idx_1])
        remaining_ids = np.array([x for x in remaining_ids if p[x] > 0])
        cand_u = num_candidates - len(idx_1)
        if len(remaining_ids) > 0:
            lg.info("No more configurations to test")
            u = np.ones(len(remaining_ids)) * (1 / len(remaining_ids))
            cand_u = min(cand_u, len(remaining_ids))
            idx_2 = np.random.choice(remaining_ids, size=cand_u, replace=False, p=u)
            idx = np.concatenate((idx_1, idx_2)).astype(int)
        else: 
            idx = np.array(idx_1).astype(int)
        new_pred_topk = [new_pred[i] for i in idx]
        return new_pred_topk

    def __make_candidates(self, configs):
        """Cfigs: A list of configuration. 

        # Each configuration is a dict, where the key is a specified 'NAME_cfg'
        and value is a point in the product space. """
        df = self.state.df_measured
        all_wl = df[df['is_wl'] == True]
        all_wl_cand = all_wl['payload_obj'].values.tolist()
        all_m = [c.module for c in all_wl_cand]
        cand_payloads = []
        lg.info("Num new candidates to be created:", len(configs))
        spec = dict(self.search_cfg)
        fn_to_name = lambda fn: fn.__name__ if fn is not None else None
        conndict = {fn_to_name(fn):fn for fn in self.connections_list}
        name_to_cls = {m.__name__:m for m in spec['modules']}
        for index, cfg in enumerate(configs):
            module_cfg = cfg['module_cfg'] # From module db
            input_cfg = cfg['input_cfg'] # From spec - required in spec
            mclsname = module_cfg['module_class']
            mkwargs = module_cfg['module_kwargs']
            input_size = input_cfg['input_shape']
            mcls = name_to_cls[mclsname]
            mkwargs = namespace_to_dict(mkwargs)
            module = mcls(**mkwargs)
            module = self.dry_run(module, input_size)
            conname = module_cfg.get('connection', None)
            fwd_fn, container = None, module
            if conname is not None:
                fwd_fn = conndict[conname]

            if fwd_fn is not None:
                container = CC(all_m, module, fwd_fn)
            else: 
                container = CC(all_m, module, fwd_noconnection)
            uid = self.__get_next_uid()
            cfg_keys = [x for x in cfg.keys() if x.endswith('_cfg')]
            pkwargs = {key: cfg[key] for key in cfg_keys}
            pkwargs['module'] = container
            pkwargs['uid'] = uid
            pkwargs['mlflow_expname'] = self.expname
            pkwargs['ignore_mixing'] = self.ignore_mixing
            payload = dict_to_namespace(pkwargs)
            cand_payloads.append(payload)
        return cand_payloads

    def dry_run(self, model, input_size):
        x = np.random.normal(size=(2,) + input_size)
        x = torch.tensor(x, dtype=torch.float32)
        _ = model(x)
        return model

    def get_wl_profile(self):
        is_wl = self.state.df_measured['is_wl'] == True
        df = self.state.df_measured[is_wl]

        reject = ['payload_obj']
        df = df[[c for c in df.columns if c not in reject]].copy()
        # Attach connection name
        return df

    def get_wl(self, round_id):
        """Return the row and payload for weak learner in round: round_id"""
        all_wl = self.state.df_measured
        all_wl = all_wl[all_wl['is_wl'] == True]
        all_wl = all_wl[all_wl['round'] == round_id]
        assert len(all_wl) == 1
        row = all_wl.iloc[0]
        return row['payload_obj'], row

    def get_ensemble(self):
        ignore_mixing = self.ignore_mixing
        pld = self.__get_ensemble_payload(ignore_mixing=ignore_mixing)
        return pld.module

    def __get_ensemble_payload(self, ignore_mixing, newpayload=None, newmweight=None):
        df = self.state.df_measured
        all_wl = df[df['is_wl'] == True]
        mixing_weights = all_wl['mixing_weight'].values.tolist()
        all_wl_cand = all_wl['payload_obj'].values.tolist()
        all_m = [c.module for c in all_wl_cand]
        if newpayload is not None:
            if newmweight is None: 
                raise ValueError("Mixing weight not provided")
            mixing_weights.append(newmweight)
            all_m.append(newpayload.module)
        if ignore_mixing is True and len(all_m) > 0:
            mixing_weights = [1.0] * len(all_m) 
        en = Ensemble(all_m, mixing_weights)
        en.eval()
        uid = self.__get_next_uid()
        # Args: uid, module,
        pkwargs = {'uid': uid, 'mlflow_expname': self.expname}
        payload = self.__modulecfgs_x_searchcfgs(pkwargs, verbose_depth=0)[0]
        payload['module'] = en
        payload = dict_to_namespace(payload)
        return payload

    def __get_next_uid(self):
        uid = self.state.pid
        self.state.pid += 1
        return uid

    def __update_profiles(self, round_id, payloads, trunk, en):
        """Updates the df-measured with new measurements."""
        epst, gammat, norm_dirtn = self.__get_step_params(payloads, trunk, en)
        step_sizes = [epst[i] / norm_dirtn[i]**2 for i in range(len(epst)) ]
        val_acc, en_val_acc, uw_en_val_acc = self.__get_accuracies(payloads, step_sizes)
        cand_infos = []
        idcol = self.idcol #.replace('-', '_')
        mcol = self.state.metric_column
        def _conn_from_pld(pld):
            try: 
                val = pld.module_cfg.connection
            except AttributeError:
                val = None
            return val

        for i, pld in enumerate(payloads):
            idval = getattr(pld.module_cfg, idcol)
            metric = getattr(pld.module_cfg, mcol)
            info = {self.idcol: idval, 'round': round_id, 'val_acc': val_acc[i],
                    'en_val_acc': en_val_acc[i], 'uw_en_val_acc':
                    uw_en_val_acc[i], 'eps_t': epst[i], 
                    'gamma_t': gammat[i], 'mixing_weight': step_sizes[i],
                    'norm_dirtn': norm_dirtn[i], 'is_wl': 0.0, mcol: metric,
                    'payload_obj': pld, 'connection': _conn_from_pld(pld)}
            mlflow.set_experiment(pld.mlflow_expname)
            with mlflow.start_run(run_id=pld.mlflow_runid):
                req = ['round', 'val_acc', 'en_val_acc', 'uw_en_val_acc', 'gamma_t', mcol, 
                       'eps_t', 'norm_dirtn', 'mixing_weight']
                info_ = {k:v for k, v in info.items() if k in req}
                mlflow.log_metrics(info_)
            cand_infos.append(info)

        cand_df = pd.DataFrame(cand_infos)
        # Drop index of candidate df
        # cand_df = cand_df.reset_index(drop=True)
        mdf = self.state.df_measured
        cols = mdf.columns
        if len(mdf) == 0:
            mdf = cand_df[cols]
        else:
            mdf = pd.concat((mdf, cand_df[cols]), axis=0)
        self.state.df_measured = mdf
        # Print new measurements
        cols = list(cols)
        cols.remove('payload_obj')
        lg.print_df(cand_df[cols], title="New measurements")
        return

    def step(self, round_id, payloads, trunk, en):
        """Updates the trained-df with new measurements."""
        self.__update_profiles(round_id, payloads, trunk, en)
        wlpayload, wlrow, reasons = self.__pick_wl(round_id)
        return wlpayload, wlrow, reasons

    def __get_step_params(self, payloads, trunk, en):
        """Returns: \eps_t, \gamma_t, \norm{f_t}
        for each payload"""
        # Returns eps_t, gamma_t, norm(f)
        ref = self.dispatch.get_step_params.remote(payloads, trunk, en)
        epst, gammat, normft = ray.get(ref)
        return epst, gammat, normft

    def __get_accuracies(self, payloads, step_sizes):
        """Returns: three lists, one each for inner_product_sum, val_acc,
        en-val-acc for each payload."""
        # for each payload, compute inner product, val-acc, en-val-acc with
        # that model added as ensemble, profile-of-that model
        en_weighted_cands = []
        en_uw_cands = []
        for i, cand in enumerate(payloads):
            en = self.__get_ensemble_payload(False, cand, step_sizes[i])
            uw_en = self.__get_ensemble_payload(True, cand, step_sizes[i])
            en_weighted_cands.append(en)
            en_uw_cands.append(uw_en)

        valacc_ref = self.dispatch.testmodel.remote(payloads, split='val')
        valacc_l = np.array(ray.get(valacc_ref))

        envalacc_ref = self.dispatch.testmodel.remote(en_weighted_cands, split='val')
        envalacc_l = np.array(ray.get(envalacc_ref))

        uw_envalacc_ref = self.dispatch.testmodel.remote(en_uw_cands, split='val')
        uw_envalacc_l = np.array(ray.get(uw_envalacc_ref))
        return valacc_l, envalacc_l, uw_envalacc_l

    # def refresh_residuals(self, ignore_mixing=True):
    #     """
    #     Computes g - h_t for the current ensemble h_t and updates the data
    #     loader to use thsi residual
    #     """
    #     # Returns a future to the residuals object
    #     en = self.get_ensemble_payload(ignore_mixing=ignore_mixing)
    #     ray.get(self.dispatch.update_residuals.remote(en))

    def train(self, *args, **kwargs):
        return ray.get(self.dispatch.train.remote(*args, **kwargs))

    def fit_and_predict(self, round_id):
        # Construct x, y
        # Fit prediction model
        # Return prediction model
        raise NotImplementedError

    def __prev_en_stats(self, round_id):
        # Returns previous ensemble latency and accuracy
        if round_id == 0: return 0.0, 0.0, 0.0
        dfm = self.state.df_measured
        dfp = self.state.df_prof
        # Get the last ensemble accuracy from latest weak learner.
        dfm = dfm[dfm['round'] == round_id - 1]
        dfm = dfm[dfm['is_wl'] == 1.0]
        assert len(dfm) == 1, dfm
        dfp = dfp[dfp.index == dfm[self.idcol].values[0]]
        metric_column = self.state.metric_column
        metric = dfp[metric_column].values[0]
        en_val_acc = dfm['en_val_acc'].values[0]
        uw_en_val_acc = dfm['uw_en_val_acc'].values[0]
        return metric, en_val_acc, uw_en_val_acc

    def __pick_wl(self, round_id, gamma_t_min=0.1):
        """Returns WL, [] if found. Else returns None, reasons"""
        df_measured = self.state.df_measured
        df_prof = self.state.df_prof
        metric_column = self.state.metric_column
        df_measured_index = df_measured['round'] == round_id
        df_round = df_measured[df_measured_index].copy()

        acc_col = 'en_val_acc'
        if self.ignore_mixing is True:
            acc_col = 'uw_en_val_acc'
        gamma_t = np.array(df_round['gamma_t'].values)
        # valacc_l = np.array(df_round['val_acc'])
        # envalacc_l = np.array(df_round['en_val_acc'])
        progress_val_acc = np.array(df_round[acc_col])
        latency_l = []
        module_ids = df_round[self.idcol]
        for _id in module_ids:
            # BUGGED
            val = df_prof.loc[_id, metric_column]
            latency_l.append(val)
        latency_l = np.array(latency_l)

        msgs = []
        # Condition 1: Make sure inner product is positive
        wlidx_innp = (gamma_t > gamma_t_min)
        # Condition 2: no-trivial progress with respect to previous ensemble
        min_del_acc = 4.0
        ret = self.__prev_en_stats(round_id)
        # lg.info("RET:", ret)
        _, prev_en_acc, prev_en_uwacc = ret[0], ret[1], ret[2]
        if acc_col == 'en_val_acc':
            prev_acc = prev_en_acc
        else:
            prev_acc = prev_en_uwacc
        progress = np.array(progress_val_acc) - prev_acc
        wlidx_progress = (progress > min_del_acc)

        wlidx = np.logical_and(wlidx_innp, wlidx_progress)
        if not np.any(wlidx_innp):
            msg = f"No candidate satisfies gamma_t condition: {gamma_t_min}"
            msgs.append(msg)
        if not np.any(wlidx_progress):
            msg = "No candidate satisfies minimum-req-en-progress."
            msgs.append(msg)
        if not np.any(wlidx):
            return None, None, msgs
        # Pick a weak learner based on minimum-latency for unit accuracy.
        wl_progress = np.zeros(len(progress))
        wl_progress[wlidx] = progress[wlidx]
        wl_progress_per_latency = wl_progress / latency_l
        _idx = np.argmax(wl_progress_per_latency)
        is_wl = [0.0] * len(df_round)
        is_wl[_idx] = 1.0
        df_round.loc[:, 'is_wl'] = is_wl
        self.state.df_measured[df_measured_index] = df_round
        payload, row = self.get_wl(round_id)
        mlflow.set_experiment(payload.mlflow_expname)
        with mlflow.start_run(run_id=payload.mlflow_runid):
            mlflow.log_metrics({'is_wl': 1.0})
        return payload, row, msgs


class CandGenQP(CandGenBase):
    """A simple linear regression model is used to fit profile vs model-args"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_key = 'uw_en_val_acc'
        self.x_keys = ['num_layers', 'blocks_list', 'emb_list', 'stride_list',
                       'round', 'connection-id']
        self.fits_list = [{'model': None, 'mean': None, 'std': None}]

    def state_dict(self):
        sd = super().state_dict()
        sd['fits_list'] = self.fits_list
        return sd

    def load_state_dict(self, sd):
        fits_list = sd['fits_list']
        self.fits_list = fits_list
        del sd['fits_list']
        super().load_state_dict(sd)

    def get_profiles(self, input_ids):
        """df_: Return classificaiton features from profile dataframe for pids in df_"""
        df_prof = self.state.df_prof
        input_rows = []
        for i, candid in enumerate(input_ids):
            row = df_prof[df_prof.index == candid]
            assert len(row) == 1,(row, candid)
            input_rows.append(row)
        x_prof = pd.concat(input_rows)
        return x_prof

    def get_features(self, input_df):
        """Converts the dataframe into features in numpy (returns df)."""
        df_prof = self.state.df_prof
        # Add a categorical variable for the connection type.
        conn_list = list(self.connections_list)
        if None in conn_list:
            conn_list.remove(None)
        cname_to_id = {fn.__name__: i + 1 for i, fn in enumerate(conn_list)}
        cname_to_id[None] = 0
        input_df['connection-id'] = input_df['connection'].map(cname_to_id)

        inp_cols = ['round', 'connection-id']
        module_cols = list(df_prof['module_kwargs'].values[0].keys())
        prof_cols = ['total_duration', 'total_params', 'flops']

        feat_list = []
        for inp_elem in input_df.to_dict('records'):
            feat_elem = {}
            # Input_DF (Measurements)
            for key, val in inp_elem.items():
                if key not in inp_cols:
                    continue
                feat_elem[key] = val
            mid = inp_elem[self.idcol]
            # Profile_DF (module args)
            prof_elem = df_prof.loc[mid].to_dict()
            for key, val in prof_elem['module_kwargs'].items():
                if key not in module_cols:
                    continue
                feat_elem[key] = val
            # Profile_DF (measurements)
            for key, val in prof_elem.items():
                if key not in prof_cols: continue
                feat_elem[key] = val
            feat_list.append(feat_elem)
        feat_df = pd.DataFrame(feat_list)
        # Convert to mean those that are list
        def to_mean(x):
            ret = x 
            try: ret = np.mean(x)
            except: pass
            return ret

        for col in feat_df.columns:
            feat_df[col] = feat_df[col].map(to_mean)

        for col in feat_df.columns:
            if col in self.x_keys : continue
            feat_df.drop(col, inplace=True, axis=1)
        return feat_df

    def fit(self):
        df_measured = self.state.df_measured
        input_df = df_measured.copy()
        # Add connection, connection-name, connection-id
        input_x = self.get_features(input_df).to_numpy()
        input_y = np.array(df_measured[self.y_key])

        mean_x = np.mean(input_x, axis=0)
        std_x = np.std(input_x, axis=0)
        std_x[std_x == 0] = 1.0
        input_x = input_x - mean_x
        input_x /= std_x 
        # input_x = input_x / np.expand_dims(std_x, axis=0)
        input_xx = np.concatenate([input_x ** 2, input_x], axis=1)
        model = LinearRegression().fit(input_xx, input_y)
        fit_out = {'model': model, 'mean': mean_x, 'std': std_x}
        self.fits_list.append(fit_out)

    def predict(self, round_id):
        fit = self.fits_list[-1]
        model, mean, std = fit['model'], fit['mean'], fit['std']
        df_prof = self.state.df_prof
        df_measured = self.state.df_measured
        input_ids = df_prof.index
        # Remove the ones that already have a measurement in the current round.
        _df = df_measured[df_measured['round'] == round_id]
        measured_ids = _df[self.idcol].values
        input_ids = [x for x in input_ids if x not in measured_ids]
        if len(input_ids) == 0:
            lg.info("All models have been measured. None to generate.")
            return None
        query_df = self.get_profiles(input_ids)
        query_df['round'] = round_id
        query_df[self.idcol] = input_ids
        conn_list = list(self.connections_list)
        if len(conn_list) == 0:
            raise ValueError("Connection list cannot be None")
        fn_to_name = lambda fn: fn.__name__ if fn is not None else None
        conn_list = [fn_to_name(f) for f in conn_list]
        df2 = pd.DataFrame({'connection': conn_list})
        query_df = query_df.merge(df2, how='cross')
        query_x = self.get_features(query_df.copy()).to_numpy()
        query_x = (query_x - mean)/std
        query_xx = np.concatenate([query_x ** 2, query_x], axis=1)
        y_pred = model.predict(query_xx)
        query_df['predicted-en-acc'] = y_pred
        return query_df

    def fit_and_predict(self, round_id):
        if round_id == 0: return None
        self.fit()
        query_df = self.predict(round_id)
        lg.info(f"Predicted df len: {len(query_df.index)}")
        return query_df

class CandGenManual(CandGenBase):
    """The original manual candidate generator."""
    def get_cand_dict(self):
        # Returns {round_id:[cands]}
        raise NotImplementedError

    def equality_test(self, a, b):
        a = np.array(a)
        b = np.array(b)
        if len(a) != len(b): return False
        return bool(np.allclose(a, b))

    def fetch_profiles(self, cands, round_id):
        """Fetch all profiles that match the keys for each canddiate
        specification"""
        cand_masks = []
        df = self.get_db()
        module_kwargs = df['module_kwargs']
        for cand in cands:
            index_mask = np.array([True] * len(df.index)).astype(bool)
            for key, val in cand.items():
                _idx = module_kwargs.map(lambda x: self.equality_test(x[key], val))
                index_mask = index_mask & _idx
                if np.sum(_idx) == 0:
                    lg.info({'msg': 'No profiles found', 'key': key, 'val': val, 'cand': cand})
                    break
            cand_masks.append(index_mask)
        if len(cand_masks) == 0:
            lg.info("No candidate profiles found!")
            return None
        cand_masks = np.stack(cand_masks)
        cand_masks = np.sum(cand_masks, axis=0)
        cand_masks = (cand_masks > 0)
        ret = df[cand_masks].copy()
        if len(ret.index) == 0:
            return None
        ret['round'] = round_id
        ret[self.idcol] = ret.index
        conn_list = list(self.connections_list)
        if len(conn_list) == 0:
            raise ValueError("Connection list cannot be None")
        fn_to_name = lambda fn: fn.__name__ if fn is not None else None
        conn_list = [fn_to_name(f) for f in conn_list]
        df2 = pd.DataFrame({'connection': [None]})
        if round_id > 0:
            df2 = pd.DataFrame({'connection': conn_list})
        ret = ret.merge(df2, how='cross')
        ret['predicted-en-acc'] = 1.0
        return ret

    def fit_and_predict(self, round_id):
        # cands = self.get_cand_dict().get(round_id, None)
        # if cands is None:
        #     lg.info(f"[red bold]No manual-candidates specified for round {round_id}")
        #     return None
        cands = self.get_cand_dict()[round_id]
        return self.fetch_profiles(cands, round_id)
