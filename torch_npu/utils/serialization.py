# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import torch
import torch_npu
import torch.nn as nn
import torch.serialization as se
from torch._six import container_abcs, string_classes

DEFAULT_PROTOCOL = 2

def to_cpu(data):
    if isinstance(data, container_abcs.Sequence):
        for i, value in enumerate(data):
            if isinstance(value, tuple):
                list_value = list(value)
                to_cpu(list_value)
                data[i] = tuple(list_value)

            elif isinstance(value, string_classes):
                continue

            elif isinstance(value, (container_abcs.Sequence, container_abcs.Mapping)):
                to_cpu(value)
            
            elif hasattr(value, "cpu"):
                data[i] = value.cpu()

    if isinstance(data, container_abcs.Mapping):
        for key, value in data.items():
            if isinstance(value, tuple):
                list_value = list(value)
                to_cpu(list_value)
                data[key] = tuple(list_value)

            elif isinstance(value, (container_abcs.Sequence, container_abcs.Mapping)):
                to_cpu(value)

            elif hasattr(value, "cpu"):
                data[key] = value.cpu()


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=False):
    """Saves the input data into a file.

    The saved data is transferred to PyTorch CPU device before being saved, so a
    following `torch.load()` will load CPU data.
    Care must be taken when working with views. Instead of saving views it's
    recommended that you recreate them after the tensors have been loaded and
    moved to their destination device(s).

    Args:
    data: The input data to be saved. Any nested combination of Python objects
        (list, tuples, sets, dicts, ...).
    path: The destination file for the data saving operation. all the writes from 
    the same host will override each other.
    """

    if isinstance(obj, torch.Tensor):
        obj = obj.cpu()
        se.save(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

    elif isinstance(obj, tuple):
        list_obj = list(obj)
        to_cpu(list_obj)
        obj = tuple(list_obj)
        se.save(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

    elif isinstance(obj, (container_abcs.Sequence, container_abcs.Mapping)):
        to_cpu(obj)
        se.save(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)
    
    elif isinstance(obj, nn.Module):
        obj = obj.cpu()
        se.save(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)
        
    else:
        se.save(obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads data previously saved with the `save()` API.

    Args:
    path (str): The path passed to the `save()` API.
    Returns:
    The loaded data.
    """
    return se.load(f, map_location, pickle_module, **pickle_load_args)
