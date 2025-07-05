import builtins
from functools import wraps

import torch
from torch.nn.parameter import UninitializedTensorMixin


def _replace_cuda_to_npu_in_list(args_list):
    for idx, arg in enumerate(args_list):
        if not isinstance(arg, builtins.bool) and isinstance(arg, builtins.int):
            args_list[idx] = f'npu:{arg}'
    return args_list


def _replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device):
    if type(device) == int:
        kwargs[device_arg] = f'npu:{device}'


def _wrapper_cuda(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        if args:
            args_new = list(args)
            args = _replace_cuda_to_npu_in_list(args_new)
        if kwargs:
            for device_arg in ["device"]:
                device = kwargs.get(device_arg, None)
                if device is not None:
                    _replace_cuda_to_npu_in_kwargs(kwargs, device_arg, device)
            device_ids = kwargs.get('device_ids', None)
            if type(device_ids) == list:
                device_ids = _replace_cuda_to_npu_in_list(device_ids)
        return fn(*args, **kwargs)

    return decorated


def _device_wrapper(enter_fn, white_list):
    for fn_name in white_list:
        fn = getattr(enter_fn, fn_name, None)
        if fn:
            setattr(enter_fn, fn_name, _wrapper_cuda(fn))


def _replace_to_method_in_allowed_methods():
    for i, method in enumerate(UninitializedTensorMixin._allowed_methods):
        if method.__name__ == "to":
            UninitializedTensorMixin._allowed_methods[i] = torch.Tensor.to
            break
