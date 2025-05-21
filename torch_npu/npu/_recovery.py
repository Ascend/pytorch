import torch
from torch.distributed.distributed_c10d import _pg_map

import torch_npu
import torch_npu._C
from torch_npu.utils._error_code import ErrCode, pta_error, _except_handler


WATCHDOG_STATUS_RUN = 1
WATCHDOG_STATUS_STOP = 2


def check_npu_storage_is_safe(storage_obj):
    if isinstance(storage_obj, (torch.storage.TypedStorage, torch.storage.UntypedStorage)):
        return torch_npu._C._check_npu_data_ptr(storage_obj)
    else:
        raise RuntimeError(f"param type should be TypedStorage or UntypedStorage, could not be {type(storage_obj)}" + pta_error(ErrCode.TYPE))


def check_npu_tensor_is_safe(tensor_obj):
    if isinstance(tensor_obj, torch.Tensor):
        return check_npu_storage_is_safe(tensor_obj.untyped_storage())
    else:
        raise RuntimeError(f"param type should be Tensor, could not be {type(tensor_obj)}" + pta_error(ErrCode.TYPE))


def mark_all_npu_tensor_unsafe(device: int):
    return torch_npu._C._mark_all_npu_data_ptr_unsafe(device)


def update_npu_storage_to_safe(storage_obj):
    if isinstance(storage_obj, (torch.storage.TypedStorage, torch.storage.UntypedStorage)):
        return torch_npu._C._update_npu_data_ptr(storage_obj)
    else:
        raise RuntimeError(f"param type should be TypedStorage or UntypedStorage, could not be {type(storage_obj)}" + pta_error(ErrCode.TYPE))


def update_npu_tensor_to_safe(tensor_obj):
    if isinstance(tensor_obj, torch.Tensor):
        return update_npu_storage_to_safe(tensor_obj.untyped_storage())
    else:
        raise RuntimeError(f"param type should be Tensor, could not be {type(tensor_obj)}" + pta_error(ErrCode.TYPE))


def set_npu_tensor_unsafe_check_flag(flag: bool) -> None:
    return torch_npu._C._set_npu_data_unsafe_flag(flag)


def get_npu_tensor_unsafe_check_flag() -> bool:
    return torch_npu._C._get_npu_data_unsafe_flag()


def _recovery_all_npu_stream(device: int) -> None:
    return torch_npu._C._recovery_all_npu_stream(device)


def restart_device(device_id: int, rebuild_all_resources: int = False):
    torch_npu.npu._lazy_init()
    if rebuild_all_resources:
        mark_all_npu_tensor_unsafe(device_id)
        set_npu_tensor_unsafe_check_flag(True)
        _recovery_all_npu_stream(device_id)
    torch_npu._C._npu_restart_device(device_id)
    _except_handler.set_force_stop_exception(False)
    # pg recovery
    npu_device = torch.device('npu')
    for pg in _pg_map:
        if (npu_device in pg._device_types):
            pg._get_backend(npu_device).clear_workmeta_list()
            pg._get_backend(npu_device).set_watchdog_status(WATCHDOG_STATUS_RUN)


def stop_device(device_id):
    torch_npu.npu._lazy_init()
    result = torch_npu._C._npu_stopDevice(device_id)
    _except_handler.set_force_stop_exception(True)
    npu_device = torch.device('npu')
    for pg in _pg_map:
        if (npu_device in pg._device_types):
            pg._get_backend(npu_device).set_watchdog_status(WATCHDOG_STATUS_STOP)
    return result
