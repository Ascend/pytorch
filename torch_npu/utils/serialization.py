import os
import pickle
import re
from typing import Any, Optional

import torch
from torch.serialization import _check_dill_version, _open_file_like, _is_zipfile, \
    _open_zipfile_reader, _is_torchscript_zip, _weights_only_unpickler, \
    _legacy_load, _load, FILE_LIKE, MAP_LOCATION, DEFAULT_PROTOCOL
from torch.serialization import _default_to_weights_only, UNSAFE_MESSAGE

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from .utils import _should_print_warning

ALWAYS_WARN_LEGACY_SERIALIZATION = False
RE_MAP_CPU = False

__all__ = ["load", "save"]


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
        if not _should_print_warning():
            return
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


def _update_cpu_remap_info(map_location):
    global RE_MAP_CPU
    RE_MAP_CPU = False
    if isinstance(map_location, (str, torch.device)) and 'cpu' in str(map_location):
        RE_MAP_CPU = True


def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: Optional[bool] = None,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any
) -> Any:
    _update_cpu_remap_info(map_location)
    torch._C._log_api_usage_once("torch.load")
    DOCS_MESSAGE = (
        "\n\nCheck the documentation of torch.load to learn more about types accepted by default with weights_only."
    )

    def _get_wo_message(message: str) -> str:
        unsafe_global_pattern = r"GLOBAL (\S+) was not an allowed global by default."
        has_unsafe_global = re.search(unsafe_global_pattern, message) is not None
        blocklist_pattern = r"whose module (\S+) is blocked"
        has_blocklist = re.search(blocklist_pattern, message) is not None
        import_pattern = r"(\S+) must be (\S+) to load"
        has_import = re.search(import_pattern, message) is not None
        if has_unsafe_global:
            updated_message = (
                "Weights only load failed. This file can still be loaded, to do so you have two options, "
                "\033[1mdo those steps only if you trust the source of the checkpoint\033[0m. "
                f"\n\t(1) {UNSAFE_MESSAGE}\n\t(2) Alternatively, to load with `weights_only=True` please check "
                "the recommended steps in the following error message.\n\tWeightsUnpickler error: "
                + message
            )
        else:
            if has_import:
                return f"Weights only load failed. {message}\n {UNSAFE_MESSAGE}\n"
            else:
                updated_message = f"Weights only load failed. {UNSAFE_MESSAGE}\n"
                if not has_blocklist:
                    updated_message += (
                        "Please file an issue with the following so that we can make "
                        "`weights_only=True` compatible with your use case: WeightsUnpickler error: "
                    )
            updated_message += message
        return updated_message + DOCS_MESSAGE

    weights_only_not_set = weights_only is None
    if weights_only_not_set:
        weights_only = _default_to_weights_only(pickle_module)

    true_values = ["1", "y", "yes", "true"]
    # Add ability to force safe only or non-safe weight loads via environment variables
    force_weights_only_load = (
        os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0") in true_values
    )
    force_no_weights_only_load = (
        os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0") in true_values
    )

    if force_weights_only_load and force_no_weights_only_load:
        raise RuntimeError(
            "Only one of `TORCH_FORCE_WEIGHTS_ONLY_LOAD` or `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` "
            "should be set, but both were set." + pta_error(ErrCode.PARAM)
        )
    elif force_weights_only_load:
        weights_only = True
    elif force_no_weights_only_load:
        # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD can only override if callsite did not explicitly set weights_only
        if weights_only_not_set:
            print(
                "Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the"
                "`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False."
            )
            weights_only = False

    if weights_only:
        if pickle_module is not None:
            raise RuntimeError("Can not safely load weights when explicit pickle_module is specified" +
                               pta_error(ErrCode.PARAM))
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
                    if weights_only:
                        raise RuntimeError(
                            "Cannot use ``weights_only=True`` with TorchScript archives passed to "
                            "``torch.load``. " + UNSAFE_MESSAGE + pta_error(ErrCode.PARAM)
                        )
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file, map_location=map_location)
                if mmap:
                    if not isinstance(f, str):
                        raise TypeError("f must be a string filename in order to use mmap argument" +
                                        pta_error(ErrCode.TYPE))
                    size = os.path.getsize(f)
                    overall_storage = torch.UntypedStorage.from_file(f, False, size)
                if weights_only:
                    try:
                        return _load(opened_zipfile, map_location, _weights_only_unpickler,
                                     overall_storage=overall_storage, **pickle_load_args)
                    except RuntimeError as e:
                        raise pickle.UnpicklingError(_get_wo_message(str(e)) + pta_error(ErrCode.SYSCALL)) from None
                return _load(opened_zipfile, map_location, pickle_module,
                             overall_storage=overall_storage, **pickle_load_args)
        else:
            if mmap:
                raise RuntimeError("mmap can only be used with files saved with "
                                   "`torch.save(_use_new_zipfile_serialization=True), "
                                   "please torch.save your checkpoint with this option in order to use mmap."
                                   + pta_error(ErrCode.PARAM))
            if weights_only:
                try:
                    return _legacy_load(opened_file, map_location, _weights_only_unpickler, **pickle_load_args)
                except RuntimeError as e:
                    raise pickle.UnpicklingError(_get_wo_message(str(e)) + pta_error(ErrCode.SYSCALL)) from None

            warn_massage = (
                "Warning: since the loaded file is not a zipfile, only \"torch.device\" and \"str\" type parameters "
                "are currently supported for parameter types of map_location. If parameter types of map_location is "
                "\"Callable[[torch.Tensor, str], torch.Tensor]\" or \"Dict[str, str]\", which is only support for "
                "zipfile, all tensors are currently loaded onto the CPU, which may introduce problems."
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


def _add_serialization_methods():
    torch.save = save
    torch.load = load
