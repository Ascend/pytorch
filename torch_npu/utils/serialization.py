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

def to_cpu(data):
    if isinstance(data, string_classes):
        return data

    if isinstance(data, torch.Tensor):
        return data.cpu()

    if isinstance(data, nn.Module):
        if torch_npu._C.is_npu(next(data.parameters())):
            setattr(data, "mark_npu", True)
        return data.cpu()

    if isinstance(data, argparse.Namespace):
        dict_obj = vars(data)
        return argparse.Namespace(**to_cpu(dict_obj))

    if isinstance(data, container_abcs.Sequence):
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
                if torch_npu._C.is_npu(next(value.parameters())):
                    setattr(value, "mark_npu", True)
                copy_data[i] = value.cpu()
            else:
                copy_data[i] = value
        return type(data)(copy_data)

    if isinstance(data, container_abcs.Mapping):
        copy_data = type(data)()
        for key, value in data.items():
            if isinstance(value, tuple):
                list_value = list(value)
                cpu_list_value = to_cpu(list_value)
                copy_data[key] = type(value)(cpu_list_value)
            elif isinstance(value, (container_abcs.Sequence, container_abcs.Mapping)):
                copy_data[key] = to_cpu(value)
            elif isinstance(value, torch.Tensor):
                copy_data[key] = value.cpu()
            elif isinstance(value, nn.Module):
                if torch_npu._C.is_npu(next(value.parameters())):
                    setattr(value, "mark_npu", True)
                copy_data[key] = value.cpu()
            else:
                copy_data[key] = value
        return copy_data

    return data


def module_to_npu(data):
    if isinstance(data, nn.Module) and hasattr(data, "mark_npu"):
        delattr(data, "mark_npu")
        data.npu()

    if isinstance(data, container_abcs.Sequence):
        for value in data:
            if isinstance(value, nn.Module) and hasattr(value, "mark_npu"):
                delattr(value, "mark_npu")
                value.npu()

    if isinstance(data, container_abcs.Mapping):
        for _, value in data.items():
            if isinstance(value, nn.Module) and hasattr(value, "mark_npu"):
                delattr(value, "mark_npu")
                value.npu()


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
    module_to_npu(obj)


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads data previously saved with the `save()` API.

    Args:
    path (str): The path passed to the `save()` API.
    Returns:
    The loaded data.
    """
    return se.load(f, 'cpu', pickle_module, **pickle_load_args)
