from functools import wraps

import torch

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


__all__ = ["npu_dtype_cast", "npu_confusion_transpose"]


def _wrap_tensor_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"torch.Tensor.{func.__name__} is deprecated and "
                           f"will be removed in future version. Use torch_npu.{func.__name__} instead." +
                           pta_error(ErrCode.NOT_SUPPORT))
    return wrapper


@_wrap_tensor_error_func
def _npu_format_cast_(self, format_or_tensor):
    return torch_npu.npu_format_cast_(self, format_or_tensor)


@_wrap_tensor_error_func
def _npu_format_cast(self, format_or_tensor):
    return torch_npu.npu_format_cast(self, format_or_tensor)


@_wrap_tensor_error_func
def npu_dtype_cast(self, dtype):
    return torch_npu.npu_dtype_cast(self, dtype)


@_wrap_tensor_error_func
def _npu_dtype_cast_(self, other):
    return torch_npu.npu_dtype_cast_(self, other)


@_wrap_tensor_error_func
def _copy_memory_(self, src, non_blocking=False):
    return torch_npu.copy_memory_(self, src, non_blocking)


@_wrap_tensor_error_func
def _one_(self):
    return torch_npu.one_(self)


@_wrap_tensor_error_func
def npu_confusion_transpose(self, perm, shape, transpose_first):
    return torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first)


def _npu(self, *args, **kwargs):
    return torch_npu._C.npu(self, *args, **kwargs)


@property
def _is_npu(self):
    return torch_npu._C.is_npu(self)


class _NPUTensortypeCache(object):
    init = False
    tensortype_list = []
    tensortype_dict = {}

    @classmethod
    def tensortype_list_dict_init(cls):
        if not cls.init:
            cls.tensortype_list += [
                torch_npu.npu.BoolTensor,
                torch_npu.npu.ByteTensor,
                torch_npu.npu.CharTensor,
                torch_npu.npu.DoubleTensor,
                torch_npu.npu.FloatTensor,
                torch_npu.npu.HalfTensor,
                torch_npu.npu.IntTensor,
                torch_npu.npu.LongTensor,
                torch_npu.npu.ShortTensor,
                torch_npu.npu.BFloat16Tensor,
            ]

            cls.tensortype_str_list = [
                "torch_npu.npu.BoolTensor",
                "torch_npu.npu.ByteTensor",
                "torch_npu.npu.CharTensor",
                "torch_npu.npu.DoubleTensor",
                "torch_npu.npu.FloatTensor",
                "torch_npu.npu.HalfTensor",
                "torch_npu.npu.IntTensor",
                "torch_npu.npu.LongTensor",
                "torch_npu.npu.ShortTensor",
                "torch_npu.npu.BFloat16Tensor",
            ]

            for tensortype, tensortype_str in zip(cls.tensortype_list, cls.tensortype_str_list):
                cls.tensortype_dict[tensortype_str] = tensortype
                cls.tensortype_dict[tensortype_str.replace('torch_npu.', 'torch.')] = tensortype

            cls.init = True

    @classmethod
    def get_tensortype_list(cls):
        return cls.tensortype_list

    @classmethod
    def get_tensortype_dict(cls):
        return cls.tensortype_dict


def _npu_type(self, dtype=None, non_blocking=False, **kwargs):
    if dtype is None:
        return self.type_raw(dtype, non_blocking, **kwargs)

    _NPUTensortypeCache.tensortype_list_dict_init()
    if isinstance(dtype, str) and dtype in _NPUTensortypeCache.get_tensortype_dict():
        tensortype_class = _NPUTensortypeCache.get_tensortype_dict()[dtype]
        return self.to(dtype=tensortype_class.dtype, device='npu', non_blocking=non_blocking)
    elif dtype in _NPUTensortypeCache.get_tensortype_list():
        return self.to(dtype=dtype.dtype, device='npu', non_blocking=non_blocking)
    else:
        return self.type_raw(dtype, non_blocking, **kwargs)


def _add_tensor_methods():
    torch.Tensor.npu_format_cast_ = _npu_format_cast_
    torch.Tensor.npu_format_cast = _npu_format_cast
    torch.Tensor.npu_dtype_cast = npu_dtype_cast
    torch.Tensor.npu_dtype_cast_ = _npu_dtype_cast_
    torch.Tensor.copy_memory_ = _copy_memory_
    torch.Tensor.one_ = _one_
    torch.Tensor.npu_confusion_transpose = npu_confusion_transpose
    torch.Tensor.type_raw = torch.Tensor.type
    torch.Tensor.type = _npu_type
