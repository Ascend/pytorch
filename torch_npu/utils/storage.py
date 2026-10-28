__all__ = []

import copy
from typing import Union
import torch

import torch_npu
from . import serialization as se


def _rebuild_npu_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, npu_storage_info):
    warn_massages = (
        "Warning: The current version of the file storing weights is old,"
        "and in the future we will deprecate the loading support for this type of file,"
        "please use 2.1 and newer torch to re-store the weight file."
    )
    se._warn_legacy_serialization(warn_massages, "oldfile")
    tensor = torch.empty(
        (0,),
        dtype=storage.dtype,
        device=storage._untyped_storage.device,
    )
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    target_device = torch.device("cpu") if se.RE_MAP_CPU else torch.device("npu")
    is_fake_mode = (
        hasattr(torch, "_guards")
        and torch._guards.detect_fake_mode(None) is not None
    )

    if is_fake_mode:
        tensor.fake_device = target_device
    elif not se.RE_MAP_CPU:
        if isinstance(npu_storage_info, bool):
            tensor = tensor.npu()
        else:
            tensor = torch_npu.npu_format_cast(tensor.npu(), npu_storage_info)
    return tensor


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type == 'npu':
        fake_tensor = torch_npu._C._tensor_construct_from_storage(self)
        return fake_tensor.cpu().untyped_storage()
    elif self.device.type != "cpu":
        return torch.UntypedStorage(self.size()).copy_(self, False)
    else:
        return self


def _deepcopy(self, memo):
    if self.device.type != 'cpu':
        # Cache the cloned UntypedStorage by _cdata (same convention as
        # UntypedStorage.__deepcopy__) and re-wrap it with self.dtype, so
        # multi-dtype views of one storage keep their own dtype.
        memo = memo.setdefault('torch', {})
        if self._cdata not in memo:
            src_tensor = torch_npu._C._tensor_construct_from_storage(self)
            # Guard against a stale or uninitialized StorageDesc dtype.
            if src_tensor.dtype != self.dtype:
                src_tensor = src_tensor.view(self.dtype)
            dst_tensor = src_tensor.clone()
            dst_tensor = torch_npu.npu_format_cast(dst_tensor, torch_npu.get_npu_format(src_tensor))
            memo[self._cdata] = dst_tensor.untyped_storage()
        return self._new_wrapped_storage(memo[self._cdata])
    else:
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))


def _share_npu_(self, *args, **kwargs):
    return torch_npu._C._share_npu_(self, *args, **kwargs)


def _typed_storage_share_npu_(self, *args, **kwargs):
    return self._untyped_storage._share_npu_(*args, **kwargs)


def _new_shared_npu(*args, **kwargs):
    return torch_npu._C._new_shared_npu(*args, **kwargs)


def _typed_storage_new_shared_npu(*args, **kwargs):
    return torch.UntypedStorage._new_shared_npu(*args, **kwargs)


def _release_ipc_counter_npu(*args, **kwargs):
    return torch_npu._C._release_ipc_counter_npu(*args, **kwargs)


def _typed_storage_release_ipc_counter_npu(*args, device: Union[str, torch.device] = "npu", **kwargs):
    return torch.UntypedStorage._release_ipc_counter_npu(*args, **kwargs)


def _add_storage_methods():
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.TypedStorage._deepcopy = _deepcopy

    torch.UntypedStorage._share_npu_ = _share_npu_
    torch.UntypedStorage._new_shared_npu = _new_shared_npu
    torch.UntypedStorage._release_ipc_counter_npu = _release_ipc_counter_npu
    torch.TypedStorage._share_npu_ = _typed_storage_share_npu_
    torch.TypedStorage._new_shared_npu = _typed_storage_new_shared_npu
    torch.TypedStorage._release_ipc_counter_npu = _typed_storage_release_ipc_counter_npu
