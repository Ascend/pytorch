import os
import io
import sys
import pickle
from typing import Any, Optional

import torch
from torch.serialization import _check_dill_version, _open_file_like, _is_zipfile, \
    _open_zipfile_reader, _is_torchscript_zip, _weights_only_unpickler, \
    _legacy_load, _load, FILE_LIKE, MAP_LOCATION, DEFAULT_PROTOCOL, \
    normalize_storage_type, location_tag, _serialization_tls

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
    weights_only: bool = False,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any
) -> Any:
    _update_cpu_remap_info(map_location)
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
                        raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e) + pta_error(ErrCode.SYSCALL)) from None
                return _load(opened_zipfile, map_location, pickle_module,
                             overall_storage=overall_storage, **pickle_load_args)
        else:
            if mmap:
                raise RuntimeError("mmap can only be used with files saved with `torch.save(_use_new_zipfile_serialization=True), ",
                                   "please torch.save your checkpoint with this option in order to use mmap." +
                                   pta_error(ErrCode.PARAM))
            if weights_only:
                try:
                    return _legacy_load(opened_file, map_location, _weights_only_unpickler, **pickle_load_args)
                except RuntimeError as e:
                    raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e) + pta_error(ErrCode.SYSCALL)) from None

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


def _npu_save(
    obj,
    zip_file,
    pickle_module,
    pickle_protocol,
    _disable_byteorder_record,
):
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
                if storage.device.type != "cpu":
                    storage_tensor = torch_npu._C._tensor_construct_from_storage(storage)
                    storage_numel = storage_tensor.size().numel() * storage_tensor.element_size() // obj._element_size()
                else:
                    storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                if storage.device.type != "cpu":
                    storage_tensor = torch_npu._C._tensor_construct_from_storage(storage)
                    storage_numel = storage_tensor.size().numel() * storage_tensor.element_size()
                else:
                    storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            if hasattr(obj, "_fake_device") and obj._fake_device is not None:
                location = str(obj._fake_device)
            else:
                location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    zip_file.write_record("data.pkl", data_value, len(data_value))

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)

        zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in sorted(serialized_storages.keys()):
        name = f"data/{key}"
        storage = serialized_storages[key]
        num_bytes = storage.nbytes()
        global _serialization_tls
        if _serialization_tls.skip_data:
            zip_file.write_record_metadata(name, num_bytes)
        else:
            # given that we copy things around anyway, we might use storage.cpu()
            # this means to that to get tensors serialized, you need to implement
            # .cpu() on the underlying Storage
            if storage.device.type != "cpu":
                storage = storage.cpu()
            # Now that it is on the CPU we can directly copy it into the zip file
            zip_file.write_record(name, storage, num_bytes)


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
    return torch.serialization.save(obj, f, pickle_module, pickle_protocol, True, _disable_byteorder_record)


def _add_serialization_methods():
    torch.save = save
    torch.load = load
    torch.serialization._save = _npu_save
