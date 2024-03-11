# Modify the dataflow to support shifts and custom augmentation.
#
from doubledistill.ddist.data import _DataFlowControl as DFC
from doubledistill.ddist.data.preprocessors import _TorchvisionTransforms
import torchvision.transforms.v2 as T
import numpy as np
import pandas as pd
import ray
import torch

class CF10CTransforms(_TorchvisionTransforms):
    def __init__(self, in_gpu_transform=True):
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        img_dims = (3, 32, 32)
        transforms_dict = {
            'train': [
                T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'val': [T.ToDtype(torch.float32), T.Normalize(mean, std)],
            'reftrain': [T.ToDtype(torch.float32), T.Normalize(mean, std)],
            'noise_val': [
                T.ColorJitter(contrast=0.5, brightness=1.0),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            
            'clip': [
                T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32),
            ],
            'noise_train':  [
                T.ColorJitter(contrast=0.5, brightness=1.0),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'noise2_test': [
                T.GaussianBlur(kernel_size=3),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'norm_transform' : [
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
        }
        super().__init__(mean, std, img_dims, in_gpu_transform,
                         transforms_dict=transforms_dict)

@ray.remote(num_gpus=0)
class AugDataFlow(DFC):
    EASY_C_SHIFTS = ['brightness', 'contrast', 'defocus_blur',
                     'elastic_transform', 'fog', 'frost', 'gaussian_blur']
    MEDIUM_C_SHIFTS = ['jpeg_compression', 'motion_blur', 'pixelate',
                       'saturate', 'snow', 'spatter', 'speckle_noise',
                       'zoom_blur']
    HARD_C_SHIFTS = ['gaussian_noise', 'glass_blur', 'impulse_noise',
                     'shot_noise']
    INET_C_SHIFTS = ['brightness', 'contrast', 'elastic_transform', 'fog',
                     'frost', 'jpeg_compression', 'pixelate', 'snow']
    ALL_C_SHIFTS = EASY_C_SHIFTS + MEDIUM_C_SHIFTS + HARD_C_SHIFTS

    def __init__(self, *args, **kwargs):
        """Overrideing the get_shard() of dataflow to allow for all kinds of
        transforms and shifts"""
        super().__init__(*args, **kwargs)
        self.shifted_shards = {}
        self.transform_gen = None

    def get_transform(self, split):
        if self.transform_gen is not None:
            return self.transform_gen.get_transform(split)
        trfngen = CF10CTransforms()
        self.transform_gen = trfngen
        return trfngen.get_transform(split)

    def get_shifted_shards(self, rank, world_size, split):
        if split in self.shifted_shards.keys():
            return self.shifted_shards[split][rank]
        read_parallelism = self.read_parallelism
        # TODO: Pick this off from ENV_VARS/CIFAR-10-C
        xarr = np.load('/data/rsaxena2/datasets/CIFAR-10-C/{}.npy'.format(split))
        yarr = np.load('/data/rsaxena2/datasets/CIFAR-10-C/labels.npy')
        # [N, H, W, C] -> [N, C, H, W
        xarr = np.transpose(xarr, (0, 3, 1, 2)).astype(np.float32)
        yarr = yarr.astype(np.float32)
        ridx = np.random.permutation(len(yarr))
        xarr, yarr = xarr[ridx], yarr[ridx]
        payload = pd.DataFrame({
            'image': [x for x in xarr], 'label': yarr, 'index':
            np.arange(len(xarr)).astype(np.int32)
        })
        split_payload = np.array_split(payload, read_parallelism)
        ds = ray.data.from_pandas(split_payload)
        norm_transform = self.get_transform('norm_transform')
        ds = ds.map(norm_transform)
        ds = ds.split(self.world_size)
        assert world_size == self.world_size
        self.shifted_shards[split] = ds
        return self.shifted_shards[split][rank]

    def getshard(self, split, rank, ddp_world_size, transform_cfg=None, device=None):
        """
        Returns a reference to the appropriate shard of the dataset.
        """
        valid_splits = ['val', 'train', 'reftrain', 'noise_val']
        valid_shifts = AugDataFlow.ALL_C_SHIFTS
        valid_splits += [f'val-shift-{s}' for s in valid_shifts]
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}")
        if ddp_world_size != self.world_size:
            raise ValueError(f"Invalid ddp_world_size: {ddp_world_size}")

        ddp_rank = rank
        if split == 'val' or 'noise_val':
            shard = self._shards_val[ddp_rank]
        elif split == 'reftrain':
            shard = self._shards_ref[ddp_rank]
        elif split == 'train':
            shard = self._shards_tr[ddp_rank]
        else:
            shard = self.get_shifted_shards(split, rank, ddp_world_size)

        transform = self.get_transform(split=split)
        if self._in_gpu_memory_ds is True:
            shard = self._getshard_gpu(split, shard, rank, ddp_world_size,
                                        device, transform, transform_cfg)
        return shard
