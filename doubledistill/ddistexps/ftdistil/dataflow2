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
            'noise_train': [
                T.ColorJitter(contrast=0.5, brightness=1.0),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'noise2_test': [
                T.GaussianBlur(kernel_size=3),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'norm_transform': [
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
        }
        super().__init__(mean, std, img_dims, in_gpu_transform,
                         transforms_dict=transforms_dict)


#@ray.remote(num_gpus=0)
class AugDataFlow(_TorchvisionTransforms):
    """ List of transformation types:
    EASY_C_SHIFTS = ['brightness', 'contrast', 'defocus_blur',
                     'elastic_transform', 'fog', 'frost', 'gaussian_blur']
    MEDIUM_C_SHIFTS = ['jpeg_compression', 'motion_blur', 'pixelate',
                       'saturate', 'snow', 'spatter', 'speckle_noise',
                       'zoom_blur']
    HARD_C_SHIFTS = ['gaussian_noise', 'glass_blur', 'impulse_noise',
                     'shot_noise']
    INET_C_SHIFTS = ['brightness', 'contrast', 'elastic_transform', 'fog',
                     'frost', 'jpeg_compression', 'pixelate', 'snow']
    ALL_C_SHIFTS = EASY_C_SHIFTS + MEDIUM_C_SHIFTS + HARD_C_SHIFTS"""

    def __init__(self, in_gpu_transform=True):
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        img_dims = (3, 32, 32)
        angle, kernel_size,sigma,max_pool_size,delta,distortion_prob, impulse_ratio,scale = 45,3,0.1,2.0,1.0, 0.05,0.05,1.0
        T.ToTensor(),
        transforms_dict = {
            'train': [
                T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'val': [T.functional.adjust_brightness(brightness_factor=(1.0)),
                    T.functional.adjust_contrast(contrast_factor=(1.0)),
                    T.functional.gaussian_blur(kernel_size=3, sigma=(3 / 3.0)),
                    T.functional.elastic_transform(grid_size=50, sigma=0.1),
                    T.functional.fog(fog_factor=(0.0, 1.0)),
                    T.functional.frost(frost_amount=(0.0, 1.0)),
                    T.functional.gaussian_noise(mean=mean, std=std),
                    T.functional.jpeg_compression(quality_factor=(0, 100)),
                    T.functional.motion_blur(kernel_size=kernel_size, angle=angle),
                    T.functional.pixelate(max_pool_size=max_pool_size),
                    T.functional.adjust_saturation(saturation_factor=(1.0 + delta)),
                    T.functional.snow(snow_amount=(0.0, 1.0)),
                    T.functional.spatter(spatter_factor=(0.0, 1.0)),
                    T.functional.speckle_noise(amount=(0.0, 1.0)),
                    T.functional.zoom_blur(scale=(1.0, 1.5)),
                    T.functional.gaussian_noise(mean=mean, std=std),
                    T.functional.glass_blur(kernel_size=kernel_size),
                    T.functional.impulse_noise(distortion_prob=distortion_prob, impulse_ratio=impulse_ratio),
                    T.functional.shot_noise(scale=scale)],
            'reftrain': [T.ToDtype(torch.float32), T.Normalize(mean, std)],
            'noise_val': [
                T.ColorJitter(contrast=0.5, brightness=1.0),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],

            'clip': [
                T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32),
            ],
            'noise_train': [
                T.ColorJitter(contrast=0.5, brightness=1.0),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'noise2_test': [
                T.GaussianBlur(kernel_size=3),
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
            'norm_transform': [
                T.ToDtype(torch.float32), T.Normalize(mean, std),
            ],
        }
        super().__init__(mean, std, img_dims, in_gpu_transform,
                         transforms_dict=transforms_dict)
