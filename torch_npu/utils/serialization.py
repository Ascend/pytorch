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
import argparse
import copy
import torch
import torch.nn as nn
import torch.serialization as se
from torch._six import container_abcs, string_classes

import torch_npu


DEFAULT_PROTOCOL = 2


def is_device(data):
    if isinstance(data, torch_npu.utils.device_guard.device):
        return True
    return False


def module_to_cpu(module):
    cpu_module = copy.deepcopy(module).cpu()
    for attr in dir(cpu_module):
        attr_item = getattr(cpu_module, attr)
        if isinstance(attr_item, torch.Tensor):
            setattr(cpu_module, attr, attr_item.cpu())
    return cpu_module


def to_cpu(data):
    if isinstance(data, string_classes):
        return data

    if isinstance(data, torch.Tensor):
        return data.cpu()

    if isinstance(data, nn.Module):
        return module_to_cpu(data)

    if isinstance(data, argparse.Namespace):
        dict_obj = vars(data)
        return argparse.Namespace(**to_cpu(dict_obj))

    if isinstance(data, container_abcs.Sequence) and not is_device(data):
        copy_data = list([None] * len(data))
        for i, value in enumerate(data):
            if isinstance(value, tuple):
                list_value = list(value)
                cpu_list_value = to_cpu(list_value)
                copy_data[i] = type(value)(cpu_list_value)
            elif isinstance(value, string_classes):
                copy_data[i] = value
            elif isinstance(value, (container_abcs.Sequence, container_abcs.Mapping)):
                copy_data[i] = to_cpu(value)
            elif isinstance(value, torch.Tensor):
                copy_data[i] = value.cpu()
            elif isinstance(value, nn.Module):
                copy_data[i] = module_to_cpu(value)
            else:
                copy_data[i] = value
        return type(data)(copy_data)

    if isinstance(data, container_abcs.Mapping):
        copy_data = type(data)()
        for key, value in data.items():
            if isinstance(value, tuple) and not is_device(value):
                list_value = list(value)
                cpu_list_value = to_cpu(list_value)
                copy_data[key] = type(value)(cpu_list_value)
            elif isinstance(value, (container_abcs.Sequence, container_abcs.Mapping)):
                copy_data[key] = to_cpu(value)
            elif isinstance(value, torch.Tensor):
                copy_data[key] = value.cpu()
            elif isinstance(value, nn.Module):
                copy_data[key] = module_to_cpu(value)
            else:
                copy_data[key] = value
        return copy_data

    return data


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
    se.save(to_cpu(obj), f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads data previously saved with the `save()` API.

    Args:
    path (str): The path passed to the `save()` API.
    Returns:
    The loaded data.
    """
    return se.load(f, 'cpu', pickle_module, **pickle_load_args)
