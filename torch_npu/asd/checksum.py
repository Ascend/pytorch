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
    
    t = 23
    c_sum = torch.sum(c, dim=-1, dtype=torch.float32)
    b1 = torch.sum(b, dim=-1, keepdim=True, dtype=torch.float32)
    c1 = matmul(a.to(torch.float32), b1)
    c1_trans = c1.squeeze(-1)

    n_b = b.shape[-1]
    m_b = b.shape[0]
    n = c.shape[-1]

    c_max, _ = torch.max(torch.abs(c), dim=-1)
    c_mean = torch.mean(torch.abs(c), dim=-1)
    c_sum_accum_error = math.sqrt(n * (n + 1) * (2 * n + 1) / 48) * c_max * 2 ** (-t)
    if torch.min(c_max / c_mean) > 5:
        c_ele_round_error_accum = c_max * 2 ** (-8) * math.sqrt(n_b)
    else:
        c_ele_round_error_accum = c_mean * 2 ** (-8) * n_b

    b_max, _ = torch.max(torch.abs(b), dim=-1, keepdim=True)
    delta_1 = math.sqrt(n_b * (n_b + 1) * (2 * n_b + 1) / 48) * b_max * 2 ** (-t)
    delta_4 = matmul(torch.abs(a), delta_1).squeeze(-1)
    a_max, _ = torch.max(torch.abs(a), dim=-1)
    delta_2_3 = math.sqrt((m_b * (m_b + 1) * (m_b + 0.5) + 2 * m_b) / 24) * a_max * torch.max(b_max) * 2 ** (-t)
    error_total = (c_sum_accum_error + c_ele_round_error_accum + delta_2_3 + delta_4).to(torch.float)

    error = torch.abs(c_sum - c1_trans)
    flag = (error - error_total) > 1e-20
    return torch.any(flag)
