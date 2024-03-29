import torch

from torch import Tensor
from torch.autograd.function import Function
from torch._subclasses.fake_tensor import FakeTensor
from torch_npu.utils.error_code import ErrCode, pta_error


class DeterministicAlgorithmsBenginOp(Function):

    @staticmethod
    def forward(ctx, tensor):
        if isinstance(tensor, FakeTensor):
            raise NotImplementedError(
                "torch.npu.enable_deterministic_algorithms do not support to graph." + pta_error(ErrCode.NOT_SUPPORT))
        else:
            with torch.autograd.profiler.record_function("deterministic_algorithms_benginop_forward"):
                torch.use_deterministic_algorithms(True)
        return tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        if isinstance(grad_outputs, FakeTensor):
            raise NotImplementedError(
                "torch.npu.enable_deterministic_algorithms do not support to graph." + pta_error(ErrCode.NOT_SUPPORT))
        else:
            with torch.autograd.profiler.record_function("deterministic_algorithms_benginop_backward"):
                torch.use_deterministic_algorithms(False)
        return grad_outputs


class DeterministicAlgorithmsEndOp(Function):
    @staticmethod
    def forward(ctx, tensor):
        if isinstance(tensor, FakeTensor):
            raise NotImplementedError(
                "torch.npu.disable_deterministic_algorithms do not support to graph." + pta_error(ErrCode.NOT_SUPPORT))
        else:
            with torch.autograd.profiler.record_function("deterministic_algorithms_endop_forward"):
                torch.use_deterministic_algorithms(False)
        return tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        if isinstance(grad_outputs, FakeTensor):
            raise NotImplementedError(
                "torch.npu.disable_deterministic_algorithms do not support to graph." + pta_error(ErrCode.NOT_SUPPORT))
        else:
            with torch.autograd.profiler.record_function("deterministic_algorithms_endop_backward"):
                torch.use_deterministic_algorithms(True)
        return grad_outputs


def enable_deterministic_with_backward(tensor: Tensor):
    return DeterministicAlgorithmsBenginOp.apply(tensor)


def disable_deterministic_with_backward(tensor: Tensor):
    return DeterministicAlgorithmsEndOp.apply(tensor)
