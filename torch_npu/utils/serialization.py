import os
import pickle
from typing import Any, Optional

import torch

from torch.serialization import _check_dill_version, _open_file_like, _is_zipfile,\
    FILE_LIKE, MAP_LOCATION, DEFAULT_PROTOCOL


def _remap_result(cpu_result, map_location):
    def traverse_dict(_dict) -> dict:
        for key,val in _dict.items():
            if isinstance(val, torch.Tensor):
                _dict[key] = val.to(map_location)
            elif isinstance(val, tuple):
                _dict[key] = traverse_tuple(val)
            elif isinstance(val, set):
                _dict[key] = traverse_set(val)
            elif isinstance(val, list):
                _dict[key] = traverse_list(val)
            elif isinstance(val, dict):
                _dict[key] = traverse_dict(val)
        return _dict

    def traverse_list(_list) -> list:
        for i, val in enumerate(_list):
            if isinstance(val, torch.Tensor):
                _list[i] = val.to(map_location)
            elif isinstance(val, tuple):
                _list[i] = traverse_tuple(val)
            elif isinstance(val, set):
                _list[i] = traverse_set(val)
            elif isinstance(val, list):
                _list[i] = traverse_list(val)
            elif isinstance(val, dict):
                _list[i] = traverse_dict(val)
        return _list

    def traverse_tuple(_tuple) -> tuple:
        new_list = []
        for val in _tuple:
            if isinstance(val, torch.Tensor):
                new_list.append(val.to(map_location))
            elif isinstance(val, tuple):
                new_list.append(traverse_tuple(val))
            elif isinstance(val, set):
                new_list.append(traverse_set(val))
            elif isinstance(val, list):
                new_list.append(traverse_list(val))
            elif isinstance(val, dict):
                new_list.append(traverse_dict(val))
            else:
                new_list.append(val)
        return tuple(new_list)

    def traverse_set(_set) -> set:
        new_list = []
        for val in iter(_set):
            if isinstance(val, torch.Tensor):
                new_list.append(val.to(map_location))
            elif isinstance(val, tuple):
                new_list.append(traverse_tuple(val))
            elif isinstance(val, set):
                new_list.append(traverse_set(val))
            elif isinstance(val, list):
                new_list.append(traverse_list(val))
            elif isinstance(val, dict):
                new_list.append(traverse_dict(val))
            else:
                new_list.append(val)
        return set(new_list)

    if isinstance(cpu_result, dict):
        return traverse_dict(cpu_result)
    elif isinstance(cpu_result, list):
        return traverse_list(cpu_result)
    elif isinstance(cpu_result, tuple):
        return traverse_tuple(cpu_result)
    elif isinstance(cpu_result, set):
        return traverse_set(cpu_result)
    elif isinstance(cpu_result, torch.Tensor):
        return cpu_result.to(map_location)
    else:
        return cpu_result


def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any
) -> Any:

    _check_dill_version(pickle_module)

    with _open_file_like(f, 'rb') as opened_file:
        if _is_zipfile(opened_file):
            return torch.serialization.load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
        else:
            print(f"Warning: since the loaded file is not a zipfile, only \"torch.device\" and \"str\" type parameters are currently supported for parameter types of map_location")
            if map_location is not None and isinstance(map_location, (torch.device, str)):
                cpu_result = torch.serialization.load(opened_file, "cpu", pickle_module, **pickle_load_args)
                if (isinstance(map_location, str) and "cpu" in map_location) or (isinstance(map_location, torch.device) and "cpu" in map_location.type):
                    return cpu_result
                return _remap_result(cpu_result, map_location)
            else:
                print(f"Warning: parameter types of map_location is \"Callable[[torch.Tensor, str], torch.Tensor]\" or \"Dict[str, str]\", which is only support for zipfile."
                               "All tensors are currently loaded onto the CPU, which may introduce problems")
                return torch.serialization.load(opened_file, "cpu", pickle_module, **pickle_load_args)


def save(
    obj: object,
    f: FILE_LIKE,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False
) -> None:
    if _use_new_zipfile_serialization is False:
        print(f"Warning: legacy save is not recommended for npu tensor, which may bring unexpected errors and hopefully set \"_use_new_zipfile_serialization = True\"",
                        "If it is necessary to use a unzipfile, convert the npu tensor to cpu tensor for saving")
    return torch.serialization.save(obj, f, pickle_module,pickle_protocol, _use_new_zipfile_serialization, _disable_byteorder_record)


def add_serialization_methods():
    torch.save = save
    torch.load = load
