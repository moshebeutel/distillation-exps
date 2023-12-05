import torch as torch

import torchvision.transforms.v2 as T


class RangeDSTransforms:
    def __init__(self):
        pass

    def get_transform(self, split, **kwargs):
        def noop(x): return x
        return noop


def default_transforms(mean, std, img_hw):
    """The default set of transforms often used for ImageNet/CIFAR training."""
    train_trfn_list = [
        T.ToDtype(torch.uint8, scale=True),
        T.RandomCrop(32, padding=4),
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
    def __init__(self, in_gpu_transform=True):
        # We will use the same for all image datasets. We'll retrain teachers if need be.
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        img_dims = (3, 32, 32)
        transforms_dict = default_transforms(mean, std, img_dims[1:])
        super().__init__(mean, std, img_dims, in_gpu_transform,
                         transforms_dict=transforms_dict)

class _ImageNetTransforms:
    def __init__(self, mean, std, img_dims, batched=True):
        self.img_dims = img_dims
        self.mean = mean
        self.std = std
        self.batched = batched
        ret = self._transforms(mean, std, img_dims, batched)
        self.process_tr, self.process_val, self.process_ref = ret

    def get_transform(self, split):
        """Returns a mapper function. (ray.data.dataset.map_batches(mapper))"""
        fn = None
        if split == 'train': fn = self.process_tr
        elif split == 'val': fn = self.process_val
        elif split == 'reftrain': fn = self.process_ref
        else: raise ValueError("Unknown split name: " + split)
        return fn

    def _to_tensorx(self, x):
        x_ = T.functional.pil_to_tensor(x).float()
        return x_

    def _transforms(self, mean, std, img_dims, batched=True):
        img_dims = img_dims[1:]
        resize_dims = img_dims
        trfn_train = T.Compose([
            T.ToPILImage(mode="RGB"),
            T.Resize(resize_dims, antialias=True),
            T.RandomResizedCrop(img_dims, antialias=True),
            T.RandomHorizontalFlip(),
            T.Lambda(self._to_tensorx),
            T.Normalize(mean, std),
        ])
        trfn_val = T.Compose([
            T.ToPILImage(mode="RGB"),
            T.Resize(resize_dims, antialias=None),
            T.CenterCrop(img_dims),
            T.Lambda(self._to_tensorx),
            T.Normalize(mean, std),
        ])
        def transform_rand(elem):
            imgbatch = elem['image']
            if batched is True:
                x = torch.stack([trfn_train(img) for img in imgbatch])
            else:
                x = trfn_train(imgbatch)
            elem['image'] = x
            return elem

        def transform_val(elem):
            imgbatch = elem['image']
            if batched is True:
                x = torch.stack([trfn_val(img) for img in imgbatch])
            else:
                x = trfn_val(imgbatch)
            elem['image'] = x
            return elem
        transform_ref = transform_val
        return transform_rand, transform_val, transform_ref


class TinyImageNetTransforms(_ImageNetTransforms):
    def __init__(self, batched=True):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        img_dims = (3, 64, 64)
        super().__init__(mean, std, img_dims, batched)

class ImageNetTransforms(_ImageNetTransforms):
    def __init__(self, batched=True):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        img_dims = (3, 224, 224)
        super().__init__(self, mean, std, img_dims, batched)


# def get_clip_transform():
#     _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
#     return preprocess

# class CIFARClipTransform:

#     def __init__(self, actor_pool=None):
#         self.__actor_pool = actor_pool
#         self.preprocess = get_clip_transform()
#         ret = self._cifar_transforms()
#         self.process_tr, self.process_val, self.process_ref = ret

#     def transform(self, ds, split, ray_remote_kwargs={}, actor_batch_size=128):
#         fn = None
#         if split == 'train': fn = self.process_tr
#         elif split == 'val': fn = self.process_val
#         elif split == 'reftrain': fn = self.process_ref
#         else: raise ValueError("Unknown split name: " + split)

#         common_kwargs = dict(ray_remote_kwargs)
#         common_kwargs['fn'] = fn
#         if self.__actor_pool is not None:
#             common_kwargs['compute'] = self.__actor_pool
#         common_kwargs['zero_copy_batch'] = True
#         common_kwargs['batch_size'] = actor_batch_size
#         common_kwargs['batch_format'] = 'pandas'
#         mapper = ds.map_batches
#         mappedds = mapper(**common_kwargs)
#         return mappedds
    
#     def _cifar_transforms(self):
#         """Img_dims in [C, H, W] format"""
#         class _apply:
#             def __init__(self, tr_fn):
#                 self.trnsfn = tr_fn

#             def __call__(self, elem):
#                 img = elem['image']
#                 label = elem['label']
#                 index = elem['index']
#                 transform_to_tensor = T.Compose([
#                     T.ToPILImage(),
#                 ])
#                 x = img.map(transform_to_tensor).map(self.trnsfn)
#                 y = label
#                 payload = pd.DataFrame({'image': x, 'label': y, 'index': index})
#                 return payload

#         transform_tr = _apply(self.preprocess)
#         transform_val = _apply(self.preprocess)
#         transform_ref = transform_val
#         return transform_tr, transform_val, transform_ref

