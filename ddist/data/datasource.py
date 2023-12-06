from typing import Optional, Tuple, List, Dict, Union, Any

import io
import os
import numpy as np
import pyarrow as pa
import pandas as pd
from PIL import Image

import ray
from ray.data import ReadTask
from ray.data.datasource.binary_datasource import BinaryDatasource
from ray.data.datasource import Datasource, Reader
from ray.data.extensions import TensorArray
from ray.data.block import Block, BlockMetadata, BlockAccessor



class ImageDatasource(BinaryDatasource):
    def __init__(self):
        super().__init__()

    def _read_file(self, f, path, include_paths, **reader_args):
        """From: ray.data.datasource.image_datasource.ImageDatasource"""
        # Secure binary record
        records = super()._read_file(f, path, include_paths=include_paths,
                                     **reader_args)
        # Cast it to images
        assert len(records) == 1
        records = records.to_pandas()
        path = records['path'].values[0]
        data = records['bytes'].values[0]
        # Convert grayscale to 3-channel if grayscale exists.
        image = Image.open(io.BytesIO(data))
        image = image.convert("RGB")
        array = np.array(image)
        # [N, H, W, C] -> [N, C, H, W
        array = np.transpose(array, (2, 0, 1))
        array = TensorArray(array)
        return pd.DataFrame({'image': [array], 'path': [path]})
