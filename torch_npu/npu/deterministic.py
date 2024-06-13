import torch

from torch import Tensor
from torch.autograd.function import Function
from torch._dynamo.decorators import forbid_in_graph


__all__ = ["enable_deterministic_with_backward", "disable_deterministic_with_backward"]


class _DeterministicAlgorithmsBeginOp(Function):

    @staticmethod
    def forward(ctx, tensor):
        with torch.autograd.profiler.record_function("deterministic_algorithms_begin_op_forward"):
            torch.use_deterministic_algorithms(True)
        return tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        with torch.autograd.profiler.record_function("deterministic_algorithms_begin_op_backward"):
            torch.use_deterministic_algorithms(False)
        return grad_outputs


class _DeterministicAlgorithmsEndOp(Function):
    @staticmethod
    def forward(ctx, tensor):
        with torch.autograd.profiler.record_function("deterministic_algorithms_end_op_forward"):
            torch.use_deterministic_algorithms(False)
        return tensor

    @staticmethod
    def backward(ctx, grad_outputs):
        with torch.autograd.profiler.record_function("deterministic_algorithms_end_op_backward"):
            torch.use_deterministic_algorithms(True)
        return grad_outputs


@forbid_in_graph
def enable_deterministic_with_backward(tensor: Tensor):
    return _DeterministicAlgorithmsBeginOp.apply(tensor)


@forbid_in_graph
def disable_deterministic_with_backward(tensor: Tensor):
    return _DeterministicAlgorithmsEndOp.apply(tensor)
