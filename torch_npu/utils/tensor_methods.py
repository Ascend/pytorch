from functools import wraps

import torch

import torch_npu


def wrap_tensor_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"torch.Tensor.{func.__name__} is deprecated and "
                           f"will be removed in future version. Use torch_npu.{func.__name__} instead.")
    return wrapper


@wrap_tensor_error_func
def npu_format_cast_(self, format_or_tensor):
    return torch_npu.npu_format_cast_(self, format_or_tensor)


@wrap_tensor_error_func
def npu_format_cast(self, format_or_tensor):
    return torch_npu.npu_format_cast(self, format_or_tensor)


@wrap_tensor_error_func
def npu_dtype_cast(self, dtype):
    return torch_npu.npu_dtype_cast(self, dtype)


@wrap_tensor_error_func
def npu_dtype_cast_(self, other):
    return torch_npu.npu_dtype_cast_(self, other)


@wrap_tensor_error_func
def copy_memory_(self, src, non_blocking=False):
    return torch_npu.copy_memory_(self, src, non_blocking)


@wrap_tensor_error_func
def one_(self):
    return torch_npu.one_(self)


@wrap_tensor_error_func
def npu_confusion_transpose(self, perm, shape, transpose_first):
    return torch_npu.npu_confusion_transpose(self, perm, shape, transpose_first)


def _npu(self, *args, **kwargs):
    return torch_npu._C.npu(self, *args, **kwargs)


@property
def _is_npu(self):
    return torch_npu._C.is_npu(self)


def add_tensor_methods():
    torch.Tensor.npu_format_cast_ = npu_format_cast_
    torch.Tensor.npu_format_cast = npu_format_cast
    torch.Tensor.npu_dtype_cast = npu_dtype_cast
    torch.Tensor.npu_dtype_cast_ = npu_dtype_cast_
    torch.Tensor.copy_memory_ = copy_memory_
    torch.Tensor.one_ = one_
    torch.Tensor.npu_confusion_transpose = npu_confusion_transpose
