__all__ = []

import math
import torch
from torch import matmul
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


def _matmul_checksum(a, b, c):
    r"""
    Compare whether there are any feature anomalies in the calculation results of matmul.
    Args:
        a(Tensor): matmul's input parameter a, and the device must be npu.
        b(Tensor): matmul's input parameter b, and the device must be npu.
        c(Tensor): matmul's output result c, and the device must be npu.

    Returns: The bool scalar tensor, located on the npu side, indicates whether there are any anomalies in the calculation result.

    """
    if not isinstance(a, torch.Tensor) or a.device.type != 'npu':
        raise TypeError(f"tensor should be torch.Tensor, and device type should be npu" + pta_error(ErrCode.PARAM))
    if not isinstance(b, torch.Tensor) or b.device.type != 'npu':
        raise TypeError(f"tensor should be torch.Tensor, and device type should be npu" + pta_error(ErrCode.PARAM))
    if not isinstance(c, torch.Tensor) or c.device.type != 'npu':
        raise TypeError(f"tensor should be torch.Tensor, and device type should be npu" + pta_error(ErrCode.PARAM))

    c_sum = torch.sum(c, dim=-1, dtype=torch.float32)
    b1 = torch.sum(b, dim=-1, keepdim=True, dtype=torch.float32)
    c1 = matmul(a.to(torch.float32), b1)
    c1_trans = c1.squeeze(-1)
    n_b = b.shape[-1]

    c_max, _ = torch.max(torch.abs(c), dim=-1)
    c_mean = torch.mean(torch.abs(c), dim=-1)
    if torch.min(c_max / c_mean) > 5:
        c_ele_round_error_accum = c_max * 2 ** (-8) * math.sqrt(n_b)
    else:
        c_ele_round_error_accum = c_mean * 2 ** (-8) * n_b

    error_total = (c_ele_round_error_accum).to(torch.float)

    error = torch.abs(c_sum - c1_trans)
    flag = (error - 5 * error_total) > 5 * 1e-20
    any_flag = torch.any(flag)
    if any_flag:
        matmul(a, b, out=c)
        c_mean2 = torch.mean(torch.abs(c), dim=-1)
        return torch.any(c_mean != c_mean2)
    return any_flag
