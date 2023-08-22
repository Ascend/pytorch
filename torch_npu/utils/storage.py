import io
import torch


def _cpu(self):
    """Returns a CPU copy of this storage if it's not already on the CPU"""
    if self.device.type != 'cpu':
        fake_tensor = torch.tensor([], device=self.device.type).set_(self)
        return fake_tensor.cpu().untyped_storage()
    else: 
        return self


def _deepcopy(self, memo):
    tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)
    return tmp_tensor._typed_storage()


def _reduce(self):
    b = io.BytesIO()
    torch.save(self, b, _use_new_zipfile_serialization=True)
    return (torch.load(io.BytesIO(b)), (b.getvalue(),))


def add_storage_methods():
    torch.storage.TypedStorage.__reduce__ = _reduce
    torch.storage._StorageBase.__reduce__ = _reduce
    torch.storage.UntypedStorage.cpu = _cpu
    torch.storage.TypedStorage._deepcopy = _deepcopy
