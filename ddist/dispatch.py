import time
import ray
from ray.experimental.tqdm_ray import tqdm
from ray.air.util.torch_dist import init_torch_dist_process_group

import torch
import torch.distributed as dist

from ddist.utils import CLog as lg
from ddist.utils import namespace_to_dict
# import tqdm from ray's extras
from ray.experimental.tqdm_ray import tqdm



def _shutdown_torch_distributed():
    """Shutdown torch distributed backend"""
    dist.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def shutdown_torch_dist_process_group(workers):
    ray.get([w.execute.remote(_shutdown_torch_distributed) for w in workers])

class BaseMapper:
    def __init__(self, worker, group_resource_req=None, n_groupworkers=1,
                 total_workers=8):
        """
        A base class that implements a map() method that maps workers to
        gpu-jobs. See: WLDispatch for usage.

        Maintains an actor pool of workers, and maps payloads to them.

        Each work function should have a signature 
            fn(rank, world_size, payload, fn_kwargs):
        """
        self.n_grpworkers = n_groupworkers
        self.total_workers = total_workers
        default = {'num_gpus': 0, 'num_cpus': 0}
        if group_resource_req is None:
            group_resource_req = default
        self.group_resource_req = group_resource_req
        self._groups = None
        self._group_is_working_on = None
        self.trainer = worker
        self._create_worker_groups()

    def _create_worker_groups(self):
        group_refs = {}
        wz =  self.n_grpworkers
        if self.total_workers % wz != 0: 
            msg = "Total number of workers should be multiple of world_size"
            raise ValueError(msg)
        num_worker_groups = self.total_workers // wz
        res_req = namespace_to_dict(self.group_resource_req)
        for grpid in range(num_worker_groups):
            grpname = f'group:{grpid}'
            group = _Group.remote(grpname, self.trainer, wz, res_req)
            group_refs[grpname] = group
        self._groups = group_refs
        self._group_is_working_on = {gkey: None for gkey in group_refs.keys()}

    def _wait(self):
        result_refs = []
        work = self._group_is_working_on
        not_ready_groups = [f for f,v in work.items() if v is not None]
        assert len(not_ready_groups) > 0
        for groupname in not_ready_groups:
            group = self._groups[groupname]
            ref = group.get_result.remote()
            result_refs.append(ref)
        ready_refs, _ = ray.wait(result_refs, num_returns=1, timeout=None)
        ret = ready_refs[0]
        groupname, payloadname, result = ray.get(ret)
        return groupname, (payloadname, result)

    def _get_ready_group(self, processed_payload_names):
        """Returns the key of the group that is ready."""
        for groupname, wrkname in self._group_is_working_on.items():
            if wrkname is None or wrkname in processed_payload_names:
                return groupname, None
        return self._wait()

    def map_workers(self, payloads, fn, fn_kwargs):
        """Will map function over jobpayload provided in `jobpayloads` list. The
        fn_kwargs will be passed on as keyword arguments.

        The fn(payload) will execute on a group of workers as specified by group
        size.

        Signature:
            fn(rank, world_size, payload, **fn_kwargs)
        Maintains order of jobpayloads
        """
        lg.info(f"Map fn:{fn.__name__} x npayloads:{len(payloads)}")
        payload_results, payload_names = {}, []
        itr_ = tqdm(enumerate(payloads), total=len(payloads), desc='Dispatch:')
        for plidx, jpl in itr_:
            payload_name = f"payload:{plidx}"
            payload_names.append(payload_name)
            grpname, fetch = self._get_ready_group(list(payload_results.keys()))
            if fetch is not None:
                self._group_is_working_on[grpname] = None
                payloadname, result = fetch
                payload_results[payloadname] = result
            wgroup = self._groups[grpname]
            _kwargs = {'fn': fn, 'fn_kwargs': fn_kwargs,
                       'payloadname': payload_name, 'payload': jpl}
            ray.get(wgroup.start.remote(**_kwargs))
            self._group_is_working_on[grpname] = payload_name
        while True:
            if len(payload_results.keys()) == len(payloads):
                break
            grpname, fetch = self._wait()
            if fetch is None:
                time.sleep(1)
                continue
            payloadname, result = fetch
            assert payloadname not in payload_results.keys()
            payload_results[payloadname] = result
            self._group_is_working_on[grpname] = None
        results = []
        for pldname in payload_names:
            results.append(payload_results[pldname])
        return results

@ray.remote
class _Group:
    def __init__(self, groupname, worker, worldsize, worker_res_req): 
        wz = worldsize
        wrefs = []
        workerfn = worker.options(**worker_res_req).remote
        for rank in range(worldsize):
            name = f'{groupname} (rk:{rank}|wz:{wz})'
            worker = workerfn(name)
            wrefs.append(worker)
        self.groupname = groupname
        self.workers = wrefs
        self.__result_refs = None
        self.__payloadname = None
        self.__result_is_ready = False

    def start(self, payload, payloadname, fn, fn_kwargs):
        ddp_mode = len(self.workers) > 1
        wgroup = self.workers
        self.__result_is_ready = False
        if ddp_mode is True:
            wz = len(wgroup)
            init_torch_dist_process_group(workers=wgroup, backend="nccl")
        # Init process group
        future_refs = []
        self.__payloadname = payloadname
        world_size = len(self.workers)
        for rank, worker in enumerate(self.workers):
            fn_kwargs['rank'] = rank
            fn_kwargs['world_size'] = world_size
            fn_kwargs['payload'] = payload 
            fut = worker.execute.remote(fn, **fn_kwargs)
            future_refs.append(fut)
        assert len(future_refs) > 0
        # Execute the function to get futures to result
        self.__result_refs = future_refs

    def get_result(self):
        """Will block."""
        if self.__result_is_ready is True:
            return self.groupname, self.__payloadname, self.__results
        vals = ray.get(self.__result_refs)
        if self.__result_refs is None:
            msg = "No results available. Call execute() first."
            raise ValueError(msg)
        # wait for all workers to finish
        vals = ray.get(self.__result_refs)
        if len(self.__result_refs) > 1:
            shutdown_torch_dist_process_group(workers=self.workers)
        self.__result_is_ready = True
        self.__results = vals
        return self.groupname, self.__payloadname, self.__results
