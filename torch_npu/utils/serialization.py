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
import io
import pickle
from typing import Any, BinaryIO, cast, Dict, Optional, Type, Tuple, Union, IO
import threading

import torch
from torch.types import Storage
import torch.serialization as se
from torch.serialization import _check_dill_version,\
    _open_zipfile_writer, location_tag, normalize_storage_type

import torch_npu

DEFAULT_PROTOCOL = 2
RE_MAP_CPU = False
save_async_stream_map = {}


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


def save_async(
    obj: object,
    f,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
    model: torch.nn.Module = None
) -> None:
    if _use_new_zipfile_serialization is False:
        raise RuntimeError("Error: torch_npu.save_async with \"_use_new_zipfile_serialization = False\"\
                           is not recommended for npu tensor, which may bring unexpected errors and hopefully \
                           set \"_use_new_zipfile_serialization = True\"",
                           "if it is necessary to use this, please convert the npu tensor to cpu tensor for saving")

    _check_dill_version(pickle_module)
    save_args = (obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization, _disable_byteorder_record)

    device = torch.npu.current_device()
    save_thread = threading.Thread(target=save_data_thread, args=(save_args, device, model))
    save_thread.start()


def save_data_thread(save_args,
                     device,
                     model: torch.nn.Module = None):
    global save_async_stream_map
    torch.npu.set_device(device)

    def hook_fn(*args):
        torch.npu.current_stream().wait_stream(save_async_stream_map.get(device))

    if device not in save_async_stream_map:
        save_async_stream = torch.npu.Stream()
        save_async_stream_map[device] = save_async_stream
        if isinstance(model, torch.nn.Module):
            model.register_full_backward_hook(hook_fn)
    else:
        save_async_stream = save_async_stream_map[device]

    obj, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization, _disable_byteorder_record = save_args
    with torch.npu.stream(save_async_stream):
        data_value, serialized_storages = _save(obj, pickle_module, pickle_protocol)
        storage_value = []
        for key in sorted(serialized_storages.keys()):
            name = f'data/{key}'
            storage = serialized_storages.get(key)
            # given that we copy things around anyway, we might use storage.cpu()
            # this means to that to get tensors serialized, you need to implement
            # .cpu() on the underlying Storage
            if storage.device.type != 'cpu':
                storage = storage.cpu()
            # Now that it is on the CPU we can directly copy it into the zip file
            num_bytes = storage.nbytes()
            storage_value.append((name, storage.data_ptr(), num_bytes))

    with _open_zipfile_writer(f) as opened_zipfile:
        opened_zipfile.write_record('data.pkl', data_value, len(data_value))

        for name, data_ptr, num_bytes in storage_value:
            opened_zipfile.write_record(name, data_ptr, num_bytes)


def _save(obj, pickle_module, pickle_protocol):
    serialized_storages = {}
    id_map: Dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj):
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):

            if isinstance(obj, torch.storage.TypedStorage):
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            'Cannot save multiple tensors or storages that '
                            'view the same data as different types')
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ('storage',
                    storage_type,
                    storage_key,
                    location,
                    storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    if isinstance(obj, torch.nn.Module):
        hook_handle = obj._backward_hooks.copy()
        obj._backward_hooks.clear()
        pickler.dump(obj)
        obj._backward_hooks.update(hook_handle)
    else:
        pickler.dump(obj)
    data_value = data_buf.getvalue()
    return data_value, serialized_storages
