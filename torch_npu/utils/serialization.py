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
import torch.serialization as se

import torch_npu

DEFAULT_PROTOCOL = 2
RE_MAP_CPU = False


def _npu_tag(obj):
    if type(obj).__module__ == 'torch_npu':
        return 'npu:' + str(obj.get_device())
    return None


def validate_npu_device(location):
    device = torch.device(str(location))
    index = device.index if device.index else 0
    if not torch_npu.npu.is_available():
        raise RuntimeError('Attempting to deserialize object on a NPU '
                           'device but torch_npu.npu.is_available() is False. '
                           'If you are running on a CPU-only machine, '
                           'please use torch.load with map_location=torch.device(\'cpu\') '
                           'to map your storages to the CPU.')
    device_count = torch_npu.npu.device_count()
    if index >= device_count:
        raise RuntimeError('Attempting to deserialize object on NPU device '
                           f'{device} but torch_npu.npu.device_count() is {device_count}. Please use '
                           'torch.load with map_location to map your storages '
                           'to an existing device.')
    return index


def _npu_deserialize(obj, location):
    if location.startswith('npu'):
        device = validate_npu_device(location)
        obj.is_npu = True
        obj.npu_index = device
    return obj 


def normalize_map_location_type(map_location):
    return str(map_location)


def update_cpu_remap_info(map_location):
    global RE_MAP_CPU
    RE_MAP_CPU = False
    if 'cpu' in map_location:
        RE_MAP_CPU = True


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True):
    """Saves the input data into a file.

    The saved data is saved by using the PyTorch CPU storage structure, but 
    following `torch.load()`  will load the corresponding NPU data.

    Care must be taken when working with views. Instead of saving views it's
    recommended that you recreate them after the tensors have been loaded and
    moved to their destination device(s).

    Args:
    data: The input data to be saved. Any nested combination of Python objects
        (list, tuples, sets, dicts, ...).
    path: The destination file for the data saving operation. all the writes from 
    the same host will override each other.
    """
    se.save(obj, f, pickle_module, pickle_protocol, True)


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads data previously saved with the `save()` API.

    Args:
    path (str): The path passed to the `save()` API.
    Returns:
    The loaded data.
    """
    map_location = normalize_map_location_type(map_location)

    update_cpu_remap_info(map_location)
    
    se._check_dill_version(pickle_module)

    if 'encoding' not in pickle_load_args.keys():
        pickle_load_args['encoding'] = 'utf-8'

    with se._open_file_like(f, 'rb') as opened_file:
        if se._is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            with se._open_zipfile_reader(opened_file) as opened_zipfile:
                if se._is_torchscript_zip(opened_zipfile):
                    warnings.warn("'torch.load' received a zip file that looks like a TorchScript archive"
                                  " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                                  " silence this warning)", UserWarning)
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file)
                return se._load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
        return se._legacy_load(opened_file, 'cpu', pickle_module, **pickle_load_args)
