import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error

__all__ = ['npu_fast_condition_index_put']


def npu_fast_condition_index_put(x, condition, value):
    """Using NPU affinity writing method to replace the native writing method in bool type index_put function.

    Examples::
    >>> x = torch.randn(128, 8192)
    >>> condition = x < 0.5
    >>> value = 0.
    >>> x1 = copy.deepcopy(x)[condition] = value
    >>> x1_opt = npu_fast_condition_index_put(x, condition, value)

    .. note::
        Because the index operator has been optimized all the time, the native implementation 
        performance of some scenarios is better.

    Args:
        x (torch.Tensor): Normal tensor.
        condition (torch.BoolTensor): Judgment condition, bool dtype.
        value (int, float): Stride of bboxes. Only IntTensor is supported.

    Returns:
        torch.Tensor: Box transformation deltas
    """

    if condition.dtype not in [torch.bool]:
        raise TypeError("Expected condition.dtype in [torch.bool]" + ops_error(ErrCode.TYPE))

    if value == 0:
        mask = torch.zeros_like(x)
    elif value == 1:
        mask = torch.ones_like(x)
    else:
        mask = torch.zeros_like(x) + value

    x = torch.where(condition, mask, x)
    return x
