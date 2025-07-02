import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error


__all__ = ["roll"]


class _RollWithIndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, index_fp, index_bp):
        N, H, W, C = input1.shape
        ctx.input1 = input1
        ctx.index_bp = index_bp
        result = input1.reshape(N, H * W, C).index_select(1, index_fp).reshape(N, H, W, C)
        return result

    @staticmethod
    def backward(ctx, grad):
        input1 = ctx.input1
        N, H, W, C = input1.shape
        index_bp = ctx.index_bp
        grad_input = grad.reshape(N, H * W, C).index_select(1, index_bp).reshape(N, H, W, C)
        return grad_input, None, None


_roll_with_index_select = _RollWithIndexSelect.apply


def _get_roll_index(H, W, shifts, device='cpu'):
    index = torch.arange(0, H * W).reshape(H, W)
    index_fp = torch.roll(index, shifts=shifts, dims=(0, 1)).reshape(-1).long()
    index_bp_dict = {i: idx for idx, i in enumerate(index_fp.numpy().tolist())}
    index_bp_list = [index_bp_dict[i] for i in range(H * W)]
    index_bp = torch.LongTensor(index_bp_list)
    return [index_fp.to(device), index_bp.to(device)]


class _NpuRollWithIndexSelect():
    """Using NPU affinity writing method to replace the native roll in swin-transformer.

    This interface is faster than the original on NPU.

    Args:
        input1 (Tensor): the input tensor.
        shifts (int or tuple of python:ints): The number of places by which the elements 
            of the tensor are shifted. If shifts is a tuple, dims must be a tuple of the 
            same size, and each dimension will be rolled by the corresponding value.
        dims (int or tuple of python:ints): Axis along which to roll

    Returns:
        Tensor: shifted input.

    Examples::
        >>> input1 = torch.randn(32, 56, 56, 16).npu()
        >>> shift_size = 3
        >>> shifted_x_npu = roll(input1, shifts=(-shift_size, -shift_size), dims=(1, 2))
    """
    def __init__(self):
        self.index_dict = {}

    def __call__(self, x, shifts, dims):
        if x.dim() != 4:
            raise ValueError("Expected x.dim() == 4" + ops_error(ErrCode.VALUE))
        if len(shifts) != 2:
            raise ValueError("Expected len(shifts) == 2" + ops_error(ErrCode.VALUE))
        if len(dims) != 2:
            raise ValueError("Expected len(dims) == 2" + ops_error(ErrCode.VALUE))
        N, H, W, C = x.shape
        key = (H, W, shifts, dims)
        if key not in self.index_dict:
            self.index_dict[key] = _get_roll_index(H, W, shifts, device=x.device)
        index_fp, index_bp = self.index_dict[key]
        return _roll_with_index_select(x, index_fp, index_bp)


roll = _NpuRollWithIndexSelect()
