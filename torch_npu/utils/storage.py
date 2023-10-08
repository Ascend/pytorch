import copy
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
    tensor = torch.tensor([], dtype=storage.dtype, device=storage.device)
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    tensor._backward_hooks = backward_hooks
    if not se.RE_MAP_CPU:
        if isinstance(npu_storage_info, bool):
            tensor = tensor.npu()
        else:
            tensor = torch_npu.npu_format_cast(tensor.npu(), npu_storage_info)
    return tensor


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch_npu._C._tensor_construct_from_storage(self)
        return fake_tensor.cpu().untyped_storage()
    else: 
        return self


def _deepcopy(self, memo):
    if self.device.type != 'cpu':
        src_tensor = torch_npu._C._tensor_construct_from_storage(self)
        dst_tensor = src_tensor.clone()
        dst_tensor = torch_npu.npu_format_cast(dst_tensor, torch_npu.get_npu_format(src_tensor))
        return dst_tensor._typed_storage()
    else:
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))


def add_storage_methods():
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.TypedStorage._deepcopy = _deepcopy
