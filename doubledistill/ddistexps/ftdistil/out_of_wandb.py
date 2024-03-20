import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from pathlib import Path
from typing import List
from functools import partial

class CF10CTransforms:
    def __init__(self):
        self.transforms_list = [
            lambda img, brightness_factor=0.5: transforms.functional.adjust_brightness(img, brightness_factor=brightness_factor),
            lambda img, contrast_factor=0.5: transforms.functional.adjust_contrast(img, contrast_factor=contrast_factor),
            lambda img, saturation_factor=0.5: transforms.functional.adjust_saturation(img, saturation_factor=saturation_factor),
            lambda img, hue_factor=0.5: transforms.functional.adjust_hue(img, hue_factor=hue_factor),
            lambda img, gamma=0.5: transforms.functional.adjust_gamma(img, gamma=gamma),
            transforms.functional.autocontrast,
            transforms.functional.equalize,
            transforms.functional.invert,
            partial(transforms.functional.posterize, bits=4),
            partial(transforms.functional.solarize, threshold=128),
            partial(transforms.functional.adjust_sharpness, sharpness_factor=0.5),
            partial(transforms.functional.rotate, degrees=15),
            partial(transforms.functional.resized_crop, size=(32, 32), scale=(0.8, 1.0)),
            partial(transforms.functional.resize, size=(32, 32)),
            partial(transforms.functional.center_crop, size=(32, 32)),
            partial(transforms.functional.gaussian_blur, kernel_size=3),
            partial(transforms.functional.perspective, distortion_scale=0.5),
            transforms.functional.to_grayscale,
            transforms.functional.to_pil_image,
            transforms.functional.to_tensor,
            transforms.functional.normalize,
            partial(transforms.functional.pad, padding=4)
        ]

    def get_transform(self, phase):
        if phase == 'train':
            return transforms.Compose([
                transform if not isinstance(transform, partial) else transform for transform in self.transforms_list
            ])
        elif phase == 'val':
            return transforms.Compose([
                transform if not isinstance(transform, partial) else transform for transform in self.transforms_list
            ])
            # return transforms.Compose([
            #     transforms.ToTensor()
            # ])
        else:
            raise ValueError(f"Unsupported phase: {phase}")

class DatasetFactory:
    DATASETS_HUB = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100}
    DATASETS_DIR = f"{str(Path.home())}/datasets/"
    NORMALIZATIONS = {'CIFAR10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      'CIFAR100': transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
    TRANSFORMS_HUB = {'CIFAR10': CF10CTransforms}

    def __init__(self, dataset_name):
        assert dataset_name in DatasetFactory.DATASETS_HUB, (f'Expected dataset name one of'
                                                             f' {DatasetFactory.DATASETS_HUB.keys()}.'
                                                             f' Got {dataset_name}')

        torchvision_transforms = DatasetFactory.TRANSFORMS_HUB[dataset_name]()
        dataset_ctor = DatasetFactory.DATASETS_HUB[dataset_name]
        dataset_dir = DatasetFactory.DATASETS_DIR + dataset_name
        print('\ntransform from get_transform: ', torchvision_transforms.get_transform('train'), '\n')
        self.train_dataset = dataset_ctor(
            root=dataset_dir,
            train=True,
            download=True,
            transform=torchvision_transforms.get_transform('train')
        )

        self.val_dataset = dataset_ctor(
            root=dataset_dir,
            train=False,
            download=True,
            transform=torchvision_transforms.get_transform('val')
        )

        print('\ntorchvision_transforms.get_transform(val',torchvision_transforms.get_transform('val'))


# Example usage:
factory = DatasetFactory('CIFAR10')
print('yes')
# Now you can access factory.train_dataset and factory.val_dataset

