__all__ = []

import copy
from typing import Any, Dict, Union
from collections import OrderedDict

import torch
from torch.storage import _warn_typed_storage_removal
from torch.overrides import has_torch_function_unary, handle_torch_function
from torch._namedtensor_internals import check_serializing_named_tensor

import torch_npu
from . import serialization as se


def _rebuild_npu_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, npu_storage_info):
    warn_massages = (
        "Warning: The current version of the file storing weights is old, "
        "and it is relanded due to internal bug of torch and compatibility issue. "
        "We will deprecate the loading support for this type of file in the future, "
        "please use newer torch to re-store the weight file."
    )
    se._warn_legacy_serialization(warn_massages, "oldfile")
    tensor = torch.tensor([], dtype=storage.dtype, device=storage._untyped_storage.device)
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    if not se.RE_MAP_CPU:
        if isinstance(npu_storage_info, bool):
            tensor = tensor.npu()
        else:
            tensor = torch_npu.npu_format_cast(tensor.npu(), npu_storage_info)
    return tensor


def _reduce_ex(self, proto):
    materialize_fake_tensors = (
        torch.serialization._serialization_tls.materialize_fake_tensors
    )
    state = torch._utils._get_obj_state(self)
    # Ignore all state when using FakeTensor with skip_data(materialize_fake_tensors) because FakeTensor has
    # some state that cannot be pickled
    if (
        # will remove hasattr, it's a hack to support versions of torch that don't have _subclasses
        hasattr(torch, "_subclasses")
        and isinstance(self, torch._subclasses.fake_tensor.FakeTensor)
        and materialize_fake_tensors
    ) or (isinstance(self, torch.Tensor) and not state):
        # For npu tensor with internal format
        check_serializing_named_tensor(self)
        torch.utils.hooks.warn_if_has_hooks(self)
        backward_hooks: Dict[Any, Any] = OrderedDict()
        if self.device.type == "npu":
            npu_storage_format = torch_npu.get_npu_format(self)
            tmp_tensor = self.cpu()
            arg_npu = (
                tmp_tensor.storage() if has_torch_function_unary(tmp_tensor) else tmp_tensor._typed_storage(),
                tmp_tensor.storage_offset(),
                tuple(tmp_tensor.size()),
                tmp_tensor.stride(),
                tmp_tensor.requires_grad,
                backward_hooks,
                npu_storage_format
            )
            return _rebuild_npu_tensor, arg_npu
        # Fast path for regular tensor without Python state.
        return self._reduce_ex_internal(proto)
    if has_torch_function_unary(self):
        return handle_torch_function(torch.Tensor.__reduce_ex__, (self,), self, proto)
    func, args = self._reduce_ex_internal(proto)
    # sizes / strides cache needs to be cleared here because it'll just be re-cached
    # if cleared earlier. Note that state references the -actual- tensor dict.
    self._clear_non_serializable_cached_data()
    return torch._tensor._rebuild_from_type_v2, (func, type(self), args, state)


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch_npu._C._tensor_construct_from_storage(self)
        return fake_tensor.cpu().untyped_storage()
    else:
        return self


def _deepcopy(self, memo):
    if self.device.type != 'cpu':
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        src_tensor = torch_npu._C._tensor_construct_from_storage(self)
        dst_tensor = src_tensor.clone()
        dst_tensor = torch_npu.npu_format_cast(dst_tensor, torch_npu.get_npu_format(src_tensor))
        new_storage = dst_tensor._typed_storage()
        memo[self._cdata] = new_storage
        return new_storage
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

    setattr(torch.UntypedStorage, "_share_npu_", _share_npu_)
    setattr(torch.UntypedStorage, "_new_shared_npu", _new_shared_npu)
    setattr(torch.UntypedStorage, "_release_ipc_counter_npu", _release_ipc_counter_npu)
    setattr(torch.TypedStorage, "_share_npu_", _typed_storage_share_npu_)
    setattr(torch.TypedStorage, "_new_shared_npu", _typed_storage_new_shared_npu)
    setattr(torch.TypedStorage, "_release_ipc_counter_npu", _typed_storage_release_ipc_counter_npu)