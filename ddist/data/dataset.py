import os
from random import sample
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import pyarrow as pa
import pandas as pd
import re

import ray
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileExtensionFilter
from ray.data.extensions import TensorArray

from ddist.utils import CLog as lg
from ddist.data.datasource import ImageDatasource

class RangeDataset():
    metadata = {'num_train_samples': 100, 'num_labels': 1, 'xshape': (3,32,32)}
    def __init__(self, xshape=None):
        if xshape is None:
            xshape = RangeDataset.metadata['xshape']
        N = self.metadata['num_train_samples']
        d = self.metadata['num_labels']
        self.xshape = xshape
        self.ds_train = self.get_data(N, d)
        self.ds_val =self.get_data(int(0.1 * N), d)

    def get_data(self, N, d):
        schema = self.get_data_schema()
        xkey, ykey = schema['x_key'], schema['y_key']
        idxkey = schema['index_key']
        xshape = (N, ) + self.xshape
        xdata = np.random.randint(low=0, high=255, size=xshape)
        xdata = xdata.astype(np.float32)
        ydata = np.random.randint(low=0, high=d, size=(N,))
        ds = {
            idxkey: np.arange(N), xkey: TensorArray(xdata),
            ykey: TensorArray(ydata),
        }
        df = pd.DataFrame(ds)
        ds = ray.data.from_pandas(df)
        return ds

    def get_split(self, split, read_parallelism=None):
        if split not in ['train', 'val']:
            raise ValueError(f"Unknown split {split}.")
        if split == 'train': return self.ds_train
        return self.ds_val

    def get_data_schema(self):
        return {'index_key': 'id', 'x_key': 'image', 'y_key': 'label'}


class _CIFARDataset():
    def __init__(self, rootdir, dsname=None, download=False,
                 num_tr_samples_per_label=None):
        """Returns images in: (3, 32, 32)"""
        download = True
        if dsname == 'CIFAR10': torchvision_ds = CIFAR10
        elif dsname == 'CIFAR100': torchvision_ds = CIFAR100
        else: raise ValueError("Unknown dataset " + self.dsname)
        self.num_tr_samples_per_label = num_tr_samples_per_label
        self.torchvision_ds_cls = torchvision_ds
        self.dsname = dsname
        self.rootdir = rootdir
        self.download = download
        self.trds, self.valds, self.refds = None, None, None

    def _setup_split(self, split, read_parallelism=None):
        if read_parallelism is None:
            read_parallelism = 32
            lg.warn("Read parallelism not provided. Using default parallelism"
                    + f" of {read_parallelism}")

        is_train = (split == 'train')
        tvds_cls = self.torchvision_ds_cls
        tvds = tvds_cls(self.rootdir, train=is_train, download=self.download)
        xarr, yarr = np.array(tvds.data), np.asarray(tvds.targets)
        # [N, H, W, C] -> [N, C, H, W
        xarr = np.transpose(xarr, (0, 3, 1, 2))
        num_ = self.num_tr_samples_per_label
        if is_train is True and num_ is not None:
            xarr, yarr = self.sample_train(xarr, yarr, num_)
        ridx = np.random.permutation(len(yarr))
        xarr, yarr = xarr[ridx], yarr[ridx]
        payload = pd.DataFrame({
            'image': [x for x in xarr], 'label': yarr, 'index':
            np.arange(len(xarr)).astype(np.int32)
        })
        # schema = BlockAccessor.batch_to_block(payload).schema
        # nparrlist = [xarr, yarr]
        split_payload = np.array_split(payload, read_parallelism)
        ds = ray.data.from_pandas(split_payload)
        if split == 'train': self.trds = ds
        elif split == 'val': self.valds = ds
        return ds

    def sample_train(self, xarr, yarr, num_tr_samples_per_label):
        # Perform sampling such that the label distribution is preserved.
        all_labels = np.unique(yarr)
        indices = []
        for lbl in all_labels:
            lbl_indices = np.where(yarr == lbl)[0]
            lbl_indices = np.random.permutation(lbl_indices)
            lbl_indices = lbl_indices[:num_tr_samples_per_label]
            indices.append(lbl_indices)
        indices = np.concatenate(indices)
        xarr, yarr = xarr[indices], yarr[indices]
        # Shuffle the data
        # Note: This is not necessary as the data is already shuffled. Howeve
        # we do this to make sure that the data is shuffled in case the
        # underlying data is not shuffled. This is a precautionary measure.
        indices = np.random.permutation(len(yarr))
        xarr, yarr = xarr[indices], yarr[indices]
        return xarr, yarr

    def get_split(self, split, read_parallelism=None):
        if split == 'train' and (self.trds is not None):
            return self.trds
        elif split == 'val' and (self.valds is not None):
            return self.valds
        elif split not in ['train', 'val']:
            raise ValueError("Unknown split " + split)
        return self._setup_split(split, read_parallelism)

    def get_data_schema(self):
        return {'x_key': 'image', 'y_key': 'label', 'index_key': 'index'}


class TinyCIFAR10Dataset(_CIFARDataset):
    """A subsampled version of CIFAR10 for debugging purposes."""
    TOT_NUM_TR_SAMPLES = 50000
    NUM_LABELS = 10
    TR_LABEL_FRAC = 0.10
    TR_SAMPLES_PER_LABEL = int((TR_LABEL_FRAC * TOT_NUM_TR_SAMPLES)/NUM_LABELS)
    NUM_TR_SAMPLES = int(TR_SAMPLES_PER_LABEL * NUM_LABELS)

    metadata = {
        'num_labels': NUM_LABELS,
        'num_train_samples': NUM_TR_SAMPLES,
        'image_dims': (3, 32, 32),
        'num_tr_samples_per_label': TR_SAMPLES_PER_LABEL,
        'tr_label_frac': TR_LABEL_FRAC,
    }

    def __init__(self, rootdir=None, download=False):
        train_fraction = TinyCIFAR10Dataset.TR_LABEL_FRAC
        lg.info(f"Using TinyCIFAR10 dataset with lable-fraction {train_fraction}")
        dsname = 'CIFAR10'
        if rootdir is None:
            _dir = os.environ['TORCH_DATA_DIR']
            rootdir = os.path.join(_dir, dsname)
        num_ = TinyCIFAR10Dataset.metadata['num_tr_samples_per_label']
        super().__init__(rootdir=rootdir, dsname=dsname, download=download,
                         num_tr_samples_per_label=num_)

class CIFAR10Dataset(_CIFARDataset):
    metadata = {
        'num_labels': 10,
        'num_train_samples':  50000,
        'image_dims': (3, 32, 32),
        'labels': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck']
    }
    def __init__(self, rootdir=None, download=False):
        dsname = 'CIFAR10'
        if rootdir is None:
            _dir = os.environ['TORCH_DATA_DIR']
            rootdir = os.path.join(_dir, dsname)
        super().__init__(rootdir=rootdir, dsname=dsname, download=download)


class CIFAR100Dataset(_CIFARDataset):
    metadata = {
        'num_labels': 100,
        'num_train_samples':  50000,
        'image_dims': (3, 32, 32),
    }
    def __init__(self, rootdir=None, download=False):
        dsname = 'CIFAR100'
        if rootdir is None:
            _dir = os.environ['TORCH_DATA_DIR']
            rootdir = os.path.join(_dir, dsname)
        super().__init__(rootdir=rootdir, dsname=dsname, download=download)


class _OnDiskImageDataset():
    def __init__(self, rootdir, file_extensions=['JPEG']):
        self.rootdir = rootdir
        self.trds, self.valds, self.refds = None, None, None
        self.trdir = os.path.join(rootdir, 'train')
        self.valdir = os.path.join(rootdir, 'val')
        self.file_extensions = file_extensions

    def get_split(self, split, read_parallelism=None):
        if split == 'train' and (self.trds is not None):
            return self.trds
        elif split == 'val' and (self.valds is not None):
            return self.valds
        elif split not in ['train', 'val']:
            raise ValueError("Unknown split " + split)
        return self._setup_split(split, read_parallelism)

    def _setup_split(self, split, parallelism):
        """parallelism: The number of ray-tasks to start. Each task will read a
        block of data.

        See: docs.ray.io/en/latest/data/performance-tips.html#tuning-read-parallelism
        """
        srcpath = self.valdir if split == 'val' else self.trdir
        lg.info("No read_parallelism provided. Using ", parallelism)
        # Construct datasets from data source. Here we need to provide the paths
        # for the reader to read.
        datasource = ImageDatasource()
        filter = FileExtensionFilter(file_extensions=self.file_extensions,
                                     allow_if_no_extension=False)
        reader_kwargs = {
            'datasource': datasource, 'paths': srcpath, 'partitioning':  None,
            'partition_filter': filter, 'filesystem': None,
            'parallelism': parallelism,
            'include_paths': True, 'ignore_missing_paths': False,
        }
        ds = ray.data.read_datasource(**reader_kwargs)

        def idx_map(elem):
            path = elem['path']
            elem['uid'] = file_to_index[path]
            try:
                elem['label'] = file_to_label[path]
            except KeyError:
                print(path)
                print(file_to_label)
                raise KeyError
            return elem

        file_paths = ds.input_files()
        file_to_index = {file: index for index, file in enumerate(file_paths)}
        file_to_label = self.file_to_labels(split, file_paths)
        ds = ds.map(idx_map)
        if split == 'train': self.trds = ds
        elif split == 'val': self.valds = ds
        elif split == 'reftrain': self.refds = ds
        return ds

    def file_to_labels(self, split, file_paths):
        labelf = os.path.join(self.rootdir, 'wnids.txt')
        with open(labelf) as f:
            labels = f.readlines()
        labels = [x.strip() for x in labels]
        labelstr_to_int = {label: index for index, label in enumerate(labels)}

        path_to_label = {}
        # if split == 'val':
        #     val_annotations = os.path.join(self.valdir, 'val_annotations.txt')
        #     df = pd.read_csv(val_annotations, sep='\t', header=None)
        #     file_names, labelstr = list(df[0]), list(df[1])
        #     filename_to_labelstr = {name: label for name, label in zip(file_names, labelstr)}
        #     for path in file_paths:
        #         labelstr = filename_to_labelstr[os.path.basename(path)]
        #         intlabel = labelstr_to_int[labelstr]
        #         path_to_label[path] = intlabel
        #         return path_to_label
         
        ptr = r'train/(.*)[/image]?'
        if split == 'val':
            ptr = r'val/(.*)[/image]?'
        for path in file_paths:
            labelstr = re.findall(ptr, path)
            assert len(labelstr) == 1, f"path:{path} ptr:{ptr} res:{labelstr}"
            labelstr = os.path.dirname(labelstr[0])
            intlabel = labelstr_to_int[labelstr]
            path_to_label[path] = intlabel
        return path_to_label



class ImageNet1kDataset(_OnDiskImageDataset):
    metadata = {
        'num_labels': 1000,
        'num_train_samples':  1281167,
        'image_dims': (3, 224, 224),
    }
    def __init__(self, rootdir, file_extensions=['JPEG']):
        if rootdir is None:
            dsname = 'ImageNet1k'
            _dir = os.environ['TORCH_DATA_DIR']
            rootdir = os.path.join(_dir, dsname)
        # Create dataset
        super().__init__(rootdir=rootdir, file_extensions=file_extensions)

    def get_split(self, split, read_parallelism=1024):
        return super().get_split(split, read_parallelism)

    def get_data_schema(self):
        return {'x_key': 'image', 'y_key': 'label', 'index_key': 'uid'}


class TinyImageNet200Dataset(_OnDiskImageDataset):
    metadata = {
        'num_labels': 200,
        'num_train_samples':  100000,
        'image_dims': (3, 64, 64),
    }
    def __init__(self, rootdir=None, file_extensions=['JPEG']):
        if rootdir is None:
            dsname = 'TinyImageNet200'
            _dir = os.environ['TORCH_DATA_DIR']
            rootdir = os.path.join(_dir, dsname)
        # Create dataset
        super().__init__(rootdir=rootdir, file_extensions=file_extensions)

    def get_split(self, split, read_parallelism=256):
        return super().get_split(split, read_parallelism)

    def get_data_schema(self):
        return {'x_key': 'image', 'y_key': 'label', 'index_key': 'uid'}
