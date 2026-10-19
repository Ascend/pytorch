__all__ = ["rebuild_npu_event", "rebuild_npu_tensor"]

import multiprocessing
import torch
from functools import wraps
from torch.multiprocessing.reductions import (  # noqa: F401
    shared_cache,
    rebuild_storage_empty,
    rebuild_storage_fd,
    StorageWeakRef,
    fd_id,
    rebuild_tensor,
    storage_from_cache,
    rebuild_meta_tensor,
    reduce_nested_tensor,
    reduce_storage,
)

import torch_npu


def rebuild_npu_event(device, handle):
    return torch.npu.Event.from_ipc_handle(device, handle)


def _npu_reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_npu_event, (event.device, handle))


def rebuild_npu_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(
            storage_cls, (storage_handle, storage_offset_bytes)
        )
        if storage is None:
            torch_npu.npu._lazy_init()
            storage = storage_cls._new_shared_npu(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(
                storage
            )
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter_npu(
                ref_counter_handle, ref_counter_offset, device=storage_device
            )

    _storage = (
        storage
        if isinstance(storage, torch.UntypedStorage)
        else storage._untyped_storage
    )

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def _npu_reduce_tensor(func):
    @wraps(func)
    def wrapper(tensor):

        reduction = func(tensor)

        rebuild_func, rebuild_args = reduction


        if rebuild_func is not rebuild_tensor:
            return reduction

        tensor_cls, storage, metadata = rebuild_args

        if storage._untyped_storage.device.type != "npu":
            return reduction

        (
            tensor_offset,
            tensor_size,
            tensor_stride,
            requires_grad,
        ) = metadata

        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_npu_()

        shared_cache[handle] = StorageWeakRef(storage)

        return (
            rebuild_npu_tensor,
            (
                tensor_cls,
                tensor_size,
                tensor_stride,
                tensor_offset,
                type(storage),
                tensor.dtype,
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )

    return wrapper


def _npu_reduce_storage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        storage = args[0] if args else kwargs.get("storage")
        if storage is not None and storage.device.type == "npu":
            raise RuntimeError(
                "Cannot pickle NPU storage; try pickling a NPU tensor instead"
            )
        return func(*args, **kwargs)
    return wrapper


def _add_reductions_methods():
    multiprocessing.reduction.register(torch.npu.Event, _npu_reduce_event)

    torch.multiprocessing.reductions.reduce_event = _npu_reduce_event

    _reduce_tensor = _npu_reduce_tensor(torch.multiprocessing.reductions.reduce_tensor)
    torch.multiprocessing.reductions.reduce_tensor = _reduce_tensor

    _reduce_storage = _npu_reduce_storage(torch.multiprocessing.reductions.reduce_storage)
    torch.multiprocessing.reductions.reduce_storage = _reduce_storage

    torch.multiprocessing.reductions.init_reductions()
