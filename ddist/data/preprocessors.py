import torch as torch

import torchvision.transforms.v2 as T


class RangeDSTransforms:
    def __init__(self, image_dims):
        pass

    def get_transform(self, split, **kwargs):
        def noop(x): return x
        return noop


def default_transforms(mean, std, img_hw):
    """The default set of transforms often used for ImageNet/CIFAR training."""
    train_trfn_list = [
        T.ToDtype(torch.uint8, scale=True),
        T.RandomCrop(img_hw, pad_if_needed=True),
        T.RandomHorizontalFlip(),
        T.ToDtype(torch.float32),
        T.Normalize(mean, std),
    ]
    val_trfn_list = [
        T.CenterCrop(img_hw),
        T.ToDtype(torch.float32),
        T.Normalize(mean, std),
    ]
    return {'train': train_trfn_list, 'val': val_trfn_list}


class _TorchvisionTransforms:
    def __init__(self, mean, std, img_dims, in_gpu_transform=True,
                 transforms_dict=None):
        self.img_dims = img_dims
        self.mean = mean
        self.std = std
        self.in_gpu_transform = in_gpu_transform
        if transforms_dict is None:
            transforms_dict = default_transforms(mean, std, img_dims[1:])

        if in_gpu_transform is True:
            import torch.nn as nn
            composer = lambda x: nn.Sequential(*x)
        else:
            composer = T.Compose
        self.transforms_dict = {k: composer(v) for k, v in transforms_dict.items()}

    def get_transform(self, split):
        """Returns a mapper function. (ray.data.dataset.map_batches(mapper))"""
        fn = self.transforms_dict.get(split, None)
        if fn is None: 
            known = list(self.transforms_dict.keys())
            raise ValueError("Unknown split name: " + split + "known: ", known)
        return fn


class BasicImageTransforms(_TorchvisionTransforms):
    def __init__(self, image_dims, in_gpu_transform=True):
        # We will use the same for all image datasets. We'll retrain teachers if need be.
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transforms_dict = default_transforms(mean, std, image_dims[1:])
        super().__init__(mean, std, image_dims, in_gpu_transform,
                         transforms_dict=transforms_dict)

