from os import read
import numpy as np
from numpy.random import sample
import ray
import torch
import pandas as pd
from ray.data.extensions import TensorArray
from ray.data import ActorPoolStrategy
from ray.util import ActorPool

from ddist.utils import get_logger
from ddist.utils import CLog as lg
from ddist.utils import Weighting as W
from ddist.data.dataset import (
    CIFAR10Dataset, CIFAR100Dataset, TinyCIFAR10Dataset,
    RangeDataset,
    TinyImageNet200Dataset, ImageNet1kDataset,
)
from ddist.data.preprocessors import (
    RangeDSTransforms,
    BasicImageTransforms,
)

def get_dataset(dataset_name):
    if dataset_name == 'RangeDS':
        return RangeDataset 
    elif dataset_name == 'CIFAR10':
        return CIFAR10Dataset
    elif dataset_name == 'TinyCIFAR10':
        return TinyCIFAR10Dataset
    elif dataset_name == 'CIFAR100':
        return CIFAR100Dataset
    elif dataset_name == 'TinyImageNet200':
        return TinyImageNet200Dataset
    elif dataset_name == 'ImageNet1k':
        return ImageNet1kDataset
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_transform(transform_name=None):
    if transform_name is None:
        return BasicImageTransforms
    if transform_name == 'RangeDS':
        return RangeDSTransforms
    elif transform_name == 'basicimage':
        return BasicImageTransforms
    raise ValueError(f"Unknown: {transform_name}")



class _DataFlowControl:
    def __init__(self, ds_name=None, ds_kwargs={}, ddp_world_size=1,
                 read_parallelism=512, keep_all_data_columns=False,
                 in_gpu_memory_ds=True):
        """
        Iterate over shards and residuals.

        The residuals are initilized to zero and need to be explicitely upated.

        datset_name: Name of the dataset to use.
        preprocessor_name: Name of the preprocessor to use.
        read_parallelism: Number of blocks the dataset will be split into.
        mapper_bz: Batch size for mapper operations.
        """
        ds_class = get_dataset(ds_name)
        ds = ds_class(**ds_kwargs)
        metadata = ds_class.metadata
        N, d = metadata['num_train_samples'], metadata['num_labels']

        self.__ds_metadata = metadata
        self.__ds_name = ds_name
        self.train_data_shape = (N, d)
        self._res_ref = ray.put(np.zeros((N, d)))
        self.__ds_schema = ds.get_data_schema()
        self.read_parallelism = read_parallelism
        # Update logk and trunk
        schema = self.__ds_schema
        xkey, ykey = schema['x_key'], schema['y_key']
        indexkey = schema['index_key']
        self.__keys_to_keep = [xkey, ykey, indexkey]
        self.__keep_all_data_columns = keep_all_data_columns
        # Create datasets
        _rp = self.read_parallelism
        ds_val = ds.get_split('val', read_parallelism=_rp)
        ds_ref = ds.get_split('train', read_parallelism=_rp)
        ds_train = ds.get_split('train', read_parallelism=_rp)
        self.ds_val, self.ds_train = ds_val, ds_train

        # Configuration
        self.world_size = ddp_world_size
        shards_val = self._create_val_shards(ds_val)
        shards_ref = self._create_reftrain_shards(ds_ref)
        shards_trn = self._create_train_shards(ds_train, None)
        self._shards_tr = shards_trn
        self._shards_val = shards_val
        self._shards_ref = shards_ref
        # Hack for in-gpu training 
        self._in_gpu_memory_ds = in_gpu_memory_ds
        if self._in_gpu_memory_ds is True:
            self._shards_tr = [shard.repartition(1) for shard in shards_trn]
            self._shards_val = [shard.repartition(1) for shard in shards_val]
            self._shards_ref = [shard.repartition(1) for shard in shards_ref]

    def get_train_shape(self):
        return self.train_data_shape

    def ready(self): 
        """Synchoronization barrier"""
        return True

    def _create_val_shards(self, ds_val):
        """Does not require randomization so we attach all mappers
        statically."""
        shards = ds_val.split(self.world_size)
        return shards

    def _create_reftrain_shards(self, ds_ref):
        """Does not require randomization so we attach all preprocessors
        statically. Does require the residuals to be updated. We leave that
        mapper to applied on each call after repeat() is called."""
        shards = ds_ref.split(self.world_size)
        return shards

    def _create_train_shards(self, ds_train, shard_bounds):
        split_bounds = None
        if shard_bounds is not None:
            split_bounds = [shard_bounds[i][1] for i in range(len(shard_bounds))]
            split_bounds = split_bounds[:-1]
        # Split to shards
        if self.world_size > 1:
            if shard_bounds is not None:
                shards = ds_train.split_at_indices(split_bounds)
            else:
                shards = ds_train.split(self.world_size)
        else: shards = [ds_train]
        return shards

    def map_select_keys(self, batch):
        # Remove unnecessary keys -- otherwise iter_torch_batches() will fail.
        if self.__keep_all_data_columns is True:
            return batch
        ret = {}
        for k in self.__keys_to_keep:
            if k in batch.keys():
                ret[k] = batch[k]
        return ret

    def attach_select_keys(self, ds):
        def select_keys(elem):
            return self.map_select_keys(elem)
        ds = ds.map(select_keys)
        return ds

    def get_data_schema(self):
        return self.__ds_schema

    def _getshard_gpu(self, split, shard, rank, ddp_world_size, device,
                       transform, transform_cfg={}):
        """The shard is converted to a dict on gpu memory and the whole shard is
        returned as a single set of tensors"""
        if device is None:
            raise ValueError("device cannot be None")
        shard = self.attach_select_keys(shard)
        numpyrefs = shard.to_numpy_refs()
        numpyobjs = ray.get(numpyrefs)
        keys = numpyobjs[0].keys()
        dataf = {k: np.concatenate([x[k] for x in numpyobjs]) for k in keys}
        datadict = {}
        for k, v in dataf.items():
            try:
                v = torch.tensor(v)
                datadict[k] = v
            except Exception as e:
                lg.info(f"Failed to convert {k} to torch.tensor v:{v}")
                raise e
        schema = self.__ds_schema
        xkey = schema['x_key']
        return _InMemDS(split, datadict, xkey, device, transform, transform_cfg)

    def getshard(self, split, rank, ddp_world_size, device=None, transform_cfg=None):
        """
        Returns a reference to the appropriate shard of the dataset.
        Also returns the preprocessor for train dataset.
        """
        if split not in ['val', 'train', 'reftrain']:
            raise ValueError(f"Invalid split: {split}")
        if ddp_world_size != self.world_size:
            raise ValueError(f"Invalid ddp_world_size: {ddp_world_size}")

        ddp_rank = rank
        shard = self._shards_tr[ddp_rank]
        tfgen = get_transform(getattr(transform_cfg, 'name', None))
        tfgen = tfgen(image_dims=self.__ds_metadata['image_dims'])
        transform = tfgen.get_transform(split='val')
        if split == 'val':
            shard = self._shards_val[ddp_rank]
        elif split == 'reftrain':
            shard = self._shards_ref[ddp_rank]
        else:
            transform = tfgen.get_transform(split='train')
            shard = self._shards_tr[ddp_rank]
        if self._in_gpu_memory_ds is True:
            shard = self._getshard_gpu(split, shard, rank, ddp_world_size,
                                        device, transform, transform_cfg)
        return shard


@ray.remote(num_gpus=0)
class DataFlowControl(_DataFlowControl):
    pass

class _InMemDS:
    def __init__(self, split, datadict, xkey, device, transform, transform_cfg): 
        self.split = split
        self.dataf = datadict
        self.transform_fn = transform.to(device)
        self.transform_cfg = transform_cfg
        self.xkey = xkey
        self.datalen = datadict[xkey].shape[0]
        self.global_shuffle_on = False
        if hasattr(transform_cfg, 'global_shuffle'):
            self.global_shuffle_on = transform_cfg.global_shuffle
        self.device = device

    def random_shuffle(self):
        self.global_shuffle_on = True
        return self

    def iter_torch_batches(self, batch_size, **kwargs):
        # shuffle data
        if self.global_shuffle_on is True:
            idx = torch.randperm(self.datalen)
        else:
            idx = torch.arange(self.datalen)
        # Move data to gpu
        for k, v in self.dataf.items():
            self.dataf[k] = v[idx].to(self.device)

        for i in range(0, self.datalen, batch_size):
            batch = {k: self.dataf[k][i:i+batch_size] for k in self.dataf.keys()}
            batch_data = batch[self.xkey]
            # Convert to float32
            trout = self.transform_fn(batch_data)
            batch[self.xkey] = trout
            yield batch
