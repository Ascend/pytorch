import torch
import torch_npu


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch.tensor([], device=self.device.type)
        fake_tensor = torch_npu._C._set_storage_with_format(fake_tensor, self)
        return fake_tensor.cpu().untyped_storage()
    else: 
        return self


def _deepcopy(self, memo):
    tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)
    return tmp_tensor._typed_storage()


def add_storage_methods():
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.TypedStorage._deepcopy = _deepcopy
