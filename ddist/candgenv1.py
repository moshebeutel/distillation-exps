import numpy as np
import pandas as pd
import itertools
import copy

import ray
from ddist.utils import CLog as lg
from ddist.models import Ensemble, ConnectionContainer as CC

# TODO: Define Payload class and do away with dict/key-value setup

class CandGeneratorBase:
    """We model/abstract out producing weak leanring candidates into a
    Generator class. The class will be provided a reference configuration after
    each round, and is expected to provide candidates for the subsequent
    round.

    base_cfg: Base reference configuation at round 0.0
    worker: A instance of wlworker class that we use to dispatch training jobs.

    To use this class, the extend_cfg() method has to be implemented.
    """
    def __init__(self, base_cfg, spec, worker):
        self.base_cfg = base_cfg
        self.wlworker = worker
        self.spec = spec

    def get_ensemble(self, all_payloads, ensemble_keys, new_mdl=None):
        all_m = []
        for round_id, elem in ensemble_keys.items():
            paykey, wl_id = elem
            # Payloads is list. Paylods of [id] is payload tuple
            mdl = all_payloads[paykey][wl_id][1]
            mdl.eval()
            all_m.append(mdl)
        if new_mdl is not None:
            all_m.append(new_mdl)
        en = Ensemble(all_m)
        en.eval()
        return en

    def get_profile(self, cand_payload, all_payloads, ensemble_keys):
        mdl_list, en_list, cfg_list = [], [], []
        for elem in cand_payload:
            model = elem[1]
            mdl_list.append(model)
            en = self.get_ensemble(all_payloads, ensemble_keys, model)
            en_list.append(en)
            cfg_list.append(elem[2])
        # for each payload, compute inner product, val-acc, en-val-acc with
        # that model added as ensemble, profile-of-that model
        innp_ref = self.wlworker.get_innp_value.remote(mdl_list)
        innp_list = ray.get(innp_ref)
        innp_sum_l = np.array([np.sum(val) for val in innp_list])
        valacc_ref = self.wlworker.testmodel.remote(mdl_list, split='val')
        valacc_l = np.array(ray.get(valacc_ref))
        envalacc_ref = self.wlworker.testmodel.remote(en_list, split='val')
        envalacc_l = np.array(ray.get(envalacc_ref))
        inp_shape = self.base_cfg['input_shape']
        flops_ref = self.wlworker.profile_model.remote(mdl_list, inp_shape)
        flops_l = ray.get(flops_ref)
        flops_l = np.array(flops_l) / 1e6
        # We profile for validation accuracy, ensemble validation accuracy,
        # innner product sum and flops for the model
        prof = {
            'val-acc': valacc_l, 'en-val-acc': envalacc_l,
            'innp-sum': innp_sum_l, 'flops(mult-accum, M)': flops_l,
        }
        profdf = pd.DataFrame(prof)
        cfgdf = pd.DataFrame(cfg_list)
        df = pd.concat([cfgdf, profdf], axis=1)
        return df

    def pick_model(self, cand_payload, all_payloads, ensemble_keys, tol):
        """Returns WL if found. Else Returns candidate with best envalacc"""
        # Get the profile for candidates
        df = self.get_profile(cand_payload, all_payloads, ensemble_keys)
        # Return None if no WL, else return model_id
        innp_sum_l = np.array(df['innp-sum'])
        valacc_l = np.array(df['val-acc'])
        envalacc_l = np.array(df['en-val-acc'])
        flops_l = np.array(df['flops(mult-accum, M)'])
        wl_id = np.argmax(envalacc_l)
        # Condition 1: Make sure inner product is positive
        wlidx_innp = (innp_sum_l > 0 - tol)
        if not np.any(wlidx_innp):
            lg.warning(f"No candidate satisfies inner product condition.{tol}")
            lg.info("Candidate details:\n", df)
            return None, df, cand_payload[wl_id]
        wlidx = wlidx_innp 
        # Condition 2: no-trivial progress with respect to previous ensemble
        min_del_acc = 4.0
        prev_en_acc, prev_en_flops = 0.0, 0.0
        if ensemble_keys or len(ensemble_keys.keys()) > 0: # is a dict
            inp_shape = self.base_cfg['input_shape']
            en = self.get_ensemble(all_payloads, ensemble_keys)
            prev_en_acc = self.wlworker.testmodel.remote([en], split='val')
            prev_en_acc = ray.get(prev_en_acc)[0]
            # lg.iinfo(f"Previous ensemble acc: {prev_en_acc}")
            prev_en_flops = self.wlworker.profile_model.remote([en], inp_shape)
            prev_en_flops = ray.get(prev_en_flops)
            prev_en_flops = np.array(prev_en_flops) / 1e6
        else:
            lg.iinfo("No previous ensemble found.")

        progress = np.array(envalacc_l) - prev_en_acc
        progress_idx = (progress > min_del_acc)
        df['change-en-acc'] = progress
        wlidx = np.logical_and(wlidx, progress_idx)
        if not np.any(wlidx):
            lg.warning("No candidate satisfies minimum-req-en-progress.")
            lg.info("Candidate details:\n", df)
            return None, df, cand_payload[wl_id]
        # Pick a weak learner based on minimum-flops spent for unit accuracy
        # gain.
        wl_progress = np.zeros(len(progress))
        wl_progress[wlidx] = progress[wlidx]
        wl_progress_per_flops = wl_progress / flops_l
        wl_id = np.argmax(wl_progress_per_flops)
        if prev_en_flops is None:
            prev_en_flops = 0.0
        wlinfo = {'wl-val-acc': valacc_l[wl_id], 
                  'en-val-acc': envalacc_l[wl_id],
                  'wl-flops': flops_l[wl_id], 
                  'en-flops': flops_l[wl_id] + prev_en_flops}
        lg.info("Candidate info:\n", df)
        lg.info("WL info:\n", wlinfo)
        wl_payload = cand_payload[wl_id]
        # return wl_id, wlinfo, wl_payload
        return wl_id, df, wl_payload

    def _get_base_candidates(self, all_payloads, ensemble_keys, refpayload):
        # Base case
        cfg, spec, payload = self.base_cfg, self.spec, []
        mclass = cfg['model_class']
        trargsl, r0argsl = spec['train_cfg'], spec['round0_cfg']
        for elem in itertools.product(trargsl, r0argsl):
            _tr, _r0 = elem
            cfg_ = copy.deepcopy(cfg)
            cfg_['model_args'] = _r0
            cfg_['train_args'] = _tr
            payload.append((_tr, mclass(**_r0), cfg_))
        return payload

    def op_add_connections(self, all_cfg, all_payloads, ensemble_keys):
        payload, spec = [], self.spec
        en = self.get_ensemble(all_payloads, ensemble_keys, new_mdl=None)
        connmap = spec['connection_cfg']
        conl = list(connmap.keys())
        trargsl = spec['train_cfg']
        for elem in itertools.product(all_cfg, trargsl, conl):
            _cfg, _tr, _cname = elem
            _cfg2 = copy.deepcopy(_cfg)
            _cfg2['connection_name'] = _cname
            mclass, margs_ = _cfg2['model_class'], _cfg2['model_args']
            mdl = mclass(**margs_)
            fwd_fn = connmap[_cname]
            cnt = CC(en, mdl, fwd_fn)
            # Payload: (train_args, model, config) : config-> model_args
            payload.append((_tr, cnt, _cfg2))
        return payload

    def extend_cfg(self, cfg):
        raise NotImplementedError

    def get_next_candidates(self, all_payloads, ensemble_keys, refpayload):
        if refpayload is None:
            return self._get_base_candidates(all_payloads, ensemble_keys,
                                             refpayload)
        # Expansion
        cfg, spec = refpayload[2], self.spec
        lg.info("Reference cfg:", cfg)
        all_cfg = self.extend_cfg(cfg)
        nconn = len(list(spec['connection_cfg'].keys()))
        lg.info(f"Attaching {nconn} connections to {len(all_cfg)} configs.")
        return self.op_add_connections(all_cfg, all_payloads, ensemble_keys)
