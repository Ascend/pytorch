import os
import pickle
from typing import Any, Optional

import torch
from torch.serialization import _check_dill_version, _open_file_like, _is_zipfile,\
    _open_zipfile_reader, _is_torchscript_zip, _weights_only_unpickler,\
    _legacy_load, _load, FILE_LIKE, MAP_LOCATION, DEFAULT_PROTOCOL

import torch_npu

ALWAYS_WARN_LEGACY_SERIALIZATION = False
RE_MAP_CPU = False


def _get_always_warn_legacy_serialization():
    return ALWAYS_WARN_LEGACY_SERIALIZATION


def _set_always_warn_legacy_serialization(always_warn: bool):
    global ALWAYS_WARN_LEGACY_SERIALIZATION
    ALWAYS_WARN_LEGACY_SERIALIZATION = always_warn


def _warn_legacy_serialization(warn_massages, key_flag: str):
    def is_first_time(flag):
        warn_key = "has_warned_for" + flag if flag else None
        if not hasattr(_warn_legacy_serialization, warn_key):
            _warn_legacy_serialization.__dict__[warn_key] = True
            return True
        else:
            return not _warn_legacy_serialization.__dict__[warn_key]

    if _get_always_warn_legacy_serialization() or is_first_time(key_flag):
        print(warn_massages)


def _remap_result(cpu_result, map_location):
    def traverse_dict(_dict) -> dict:
        for key, val in _dict.items():
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


def update_cpu_remap_info(map_location):
    global RE_MAP_CPU
    RE_MAP_CPU = False
    if isinstance(map_location, (str, torch.device)) and 'cpu' in str(map_location):
        RE_MAP_CPU = True


def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any
) -> Any:
    update_cpu_remap_info(map_location)
    torch._C._log_api_usage_once("torch.load")
    UNSAFE_MESSAGE = (
        "Weights only load failed. Re-running `torch.load` with `weights_only` set to `False`"
        " will likely succeed, but it can result in arbitrary code execution."
        "Do it only if you get the file from a trusted source. WeightsUnpickler error: "
    )
    # Add ability to force safe only weight loads via environment variable
    if os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0").lower() in ['1', 'y', 'yes', 'true']:
        weights_only = True

    if weights_only:
        if pickle_module is not None:
            raise RuntimeError("Can not safely load weights when explicit pickle_module is specified")
    else:
        if pickle_module is None:
            pickle_module = pickle

    # make flipping default BC-compatible
    if mmap is None:
        mmap = False
    _check_dill_version(pickle_module)

    if 'encoding' not in pickle_load_args.keys():
        pickle_load_args['encoding'] = 'utf-8'

    with _open_file_like(f, 'rb') as opened_file:
        if _is_zipfile(opened_file):
            orig_position = opened_file.tell()
            overall_storage = None
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    print(f"Warning: 'torch.load' received a zip file that looks like a TorchScript archive"
                          " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to silence this warning)")
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file, map_location=map_location)
                if mmap:
                    if not isinstance(f, str):
                        raise ValueError("f must be a string filename in order to use mmap argument")
                    size = os.path.getsize(f)
                    overall_storage = torch.UntypedStorage.from_file(f, False, size)
                if weights_only:
                    try:
                        return _load(opened_zipfile, map_location, _weights_only_unpickler,
                                     overall_storage=overall_storage, **pickle_load_args)
                    except RuntimeError as e:
                        raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None
                return _load(opened_zipfile, map_location, pickle_module,
                             overall_storage=overall_storage, **pickle_load_args)
        else:
            if mmap:
                raise RuntimeError("mmap can only be used with files saved with `torch.save(_use_new_zipfile_serialization=True), ",
                                   "please torch.save your checkpoint with this option in order to use mmap.")
            if weights_only:
                try:
                    return _legacy_load(opened_file, map_location, _weights_only_unpickler, **pickle_load_args)
                except RuntimeError as e:
                    raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None

            warn_massage = (
                "Warning: since the loaded file is not a zipfile, only \"torch.device\" and \"str\" type parameters are currently supported for parameter types of map_location"
                "If parameter types of map_location is \"Callable[[torch.Tensor, str], torch.Tensor]\" or \"Dict[str, str]\", which is only support for zipfile,"
                "all tensors are currently loaded onto the CPU, which may introduce problems"
            )
            _warn_legacy_serialization(warn_massage, "load")

            if map_location is not None and isinstance(map_location, (torch.device, str)):
                cpu_result = _legacy_load(opened_file, "cpu", pickle_module, **pickle_load_args)
                if isinstance(map_location, str) and "cpu" in map_location:
                    return cpu_result
                if isinstance(map_location, torch.device) and "cpu" in map_location.type:
                    return cpu_result
                return _remap_result(cpu_result, map_location)
            else:
                return _legacy_load(opened_file, "cpu", pickle_module, **pickle_load_args)


def _get_npu_save_result(
    obj: object,
    f: FILE_LIKE,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False
) -> None:
    cpu_nbytes = torch.storage.UntypedStorage.nbytes

    def npu_nbytes(self):
        if self.device.type != 'cpu':
            storage_tensor = torch_npu._C._tensor_construct_from_storage(self)
            base_nbytes = storage_tensor.size().numel() * storage_tensor.element_size()
            return base_nbytes
        else:
            return cpu_nbytes(self)

    torch.storage.UntypedStorage.nbytes = npu_nbytes
    result = torch.serialization.save(obj, f, pickle_module, pickle_protocol, True, _disable_byteorder_record)
    torch.storage.UntypedStorage.nbytes = cpu_nbytes
    return result


def save(
    obj: object,
    f: FILE_LIKE,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False
) -> None:
    if _use_new_zipfile_serialization is False:
        warn_massage = (
            "Warning: torch.save with \"_use_new_zipfile_serialization = False\" is not recommended for npu tensor, which may bring unexpected errors and hopefully set \"_use_new_zipfile_serialization = True\"",
            "if it is necessary to use this, please convert the npu tensor to cpu tensor for saving"
        )
        _warn_legacy_serialization(warn_massage, "save")
    return _get_npu_save_result(obj, f, pickle_module, pickle_protocol, True, _disable_byteorder_record)


def add_serialization_methods():
    torch.save = save
    torch.load = load
