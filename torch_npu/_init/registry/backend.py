import torch

import torch_npu


def register_privateuse1_backend():
    torch.utils.rename_privateuse1_backend("npu")
    # rename device name to 'npu' and register funcs
    torch._register_device_module("npu", torch_npu.npu)
    unsupported_dtype = [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]
    torch_npu.unsupported_dtype = unsupported_dtype
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True,
        for_module=True,
        for_storage=True,
        unsupported_dtype=unsupported_dtype,
    )
    torch.nn.parameter.UninitializedTensorMixin._allowed_methods.append(
        torch.Tensor.npu
    )
