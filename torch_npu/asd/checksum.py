__all__ = []

import math
import torch
from torch import matmul
from torch_npu.utils._error_code import ErrCode, pta_error

# _npu_matmul_abft_verify (V-ABFT) integration: aclnnMatmulAbftVerify checks a
# precomputed matmul result with block checksums and a variance-adaptive
# threshold. Its symbols only exist in CANN environments that ship the
# operator, so matmul_checksum falls back to pure PyTorch when it is unusable.

# Supported dtypes and per-dtype recommended e_max, per the operator doc
# (FP16 has no recommended e_max and stays unsupported).
_ABFT_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
_ABFT_DEFAULT_EMAX = {
    torch.bfloat16: 0.001,
    torch.float32: 0.00002,
}
_ABFT_ROW_PACK = 8        # row verdicts packed per comp_row byte
_ABFT_ONES_CACHE_LIMIT = 64

_abft_available_cache = None  # None: not probed yet, otherwise bool
_abft_ones_cache = {}         # (n, dtype, device) -> checksum weight tensor
_abft_mask_cache = {}         # (m, numel, device) -> valid-bit mask for comp_row


def _npu_matmul_abft_verify_available():
    r"""Probe once whether ``torch.ops.npu._npu_matmul_abft_verify`` is usable.

    Fails when the op is not registered or the runtime CANN cannot resolve the
    ``aclnnMatmulAbftVerify`` symbols (EXEC_NPU_CMD dlopen/dlsym at call
    time); neither can change within the process, so the result is cached.
    """
    global _abft_available_cache
    if _abft_available_cache is None:
        try:
            probe_a = torch.ones(_ABFT_ROW_PACK, 16, dtype=torch.bfloat16, device="npu")
            probe_b = torch.ones(16, 16, dtype=torch.bfloat16, device="npu")
            probe_c = torch.full((_ABFT_ROW_PACK, 16), 16.0, dtype=torch.float32, device="npu")
            probe_w = torch.ones(16, dtype=torch.bfloat16, device="npu")
            torch.ops.npu._npu_matmul_abft_verify(probe_a, probe_b, probe_c, probe_w, e_max=0.001)
            _abft_available_cache = True
        except (RuntimeError, AttributeError, TypeError):
            _abft_available_cache = False
    return _abft_available_cache


def _abft_ones_weight(n, dtype, device):
    r"""Cached all-ones checksum weight of shape [n]; all-ones selects the
    plain block row-checksum verification (operator-recommended usage).
    """
    key = (n, dtype, device)
    weight = _abft_ones_cache.get(key)
    if weight is None:
        if len(_abft_ones_cache) >= _ABFT_ONES_CACHE_LIMIT:
            _abft_ones_cache.clear()
        weight = torch.ones(n, dtype=dtype, device=device)
        _abft_ones_cache[key] = weight
    return weight


def _abft_comp_row_to_flag(comp_row, m):
    r"""Reduce the packed verdict bitstream ``comp_row`` to one bool
    (True = error detected).

    Layout: byte ``seg * ceil(M/8) + rg`` holds the verdicts of rows
    ``rg*8 .. rg*8+7`` in column segment ``seg``, LSB first; a set bit means
    correct. Padding bits of rows >= M are undefined and must be masked out
    (mask cached by (m, numel, device)).
    """
    if m % _ABFT_ROW_PACK == 0:
        return torch.any(comp_row != 255)
    key = (m, comp_row.numel(), comp_row.device)
    mask = _abft_mask_cache.get(key)
    if mask is None:
        num_row_groups = (m + _ABFT_ROW_PACK - 1) // _ABFT_ROW_PACK
        mask = torch.full((comp_row.numel(),), 255, dtype=torch.uint8, device=comp_row.device)
        mask[num_row_groups - 1::num_row_groups] = (1 << (m % _ABFT_ROW_PACK)) - 1
        if len(_abft_mask_cache) >= _ABFT_ONES_CACHE_LIMIT:
            _abft_mask_cache.clear()
        _abft_mask_cache[key] = mask
    return torch.any((comp_row & mask) != mask)


def _abft_path_applicable(a, b, c):
    r"""Whether (a, b, c) fits the op contract: 2D, a/b of the same bf16/fp32
    dtype, c of a's dtype or fp32, consistent non-degenerate shapes.
    Everything else keeps the legacy PyTorch path.
    """
    if a.dim() != 2 or b.dim() != 2 or c.dim() != 2:
        return False
    if a.dtype != b.dtype or a.dtype not in _ABFT_SUPPORTED_DTYPES:
        return False
    if c.dtype != a.dtype and c.dtype != torch.float32:
        return False
    if a.shape[0] == 0 or a.shape[1] == 0 or b.shape[1] == 0:
        return False
    if a.shape[1] != b.shape[0] or c.shape[0] != a.shape[0] or c.shape[1] != b.shape[1]:
        return False
    return True


def _matmul_checksum_abft(a, b, c):
    r"""V-ABFT optimized path via ``torch.ops.npu._npu_matmul_abft_verify``."""
    e_max = _ABFT_DEFAULT_EMAX.get(a.dtype, 0.001)
    weight = _abft_ones_weight(b.shape[1], a.dtype, a.device)
    c_fp32 = c.to(torch.float32) if c.dtype != torch.float32 else c
    comp_row = torch.ops.npu._npu_matmul_abft_verify(
        a.contiguous(), b.contiguous(), c_fp32.contiguous(), weight, e_max=e_max)
    return _abft_comp_row_to_flag(comp_row, a.shape[0])


def _check_matmul_checksum_inputs(a, b, c):
    if not isinstance(a, torch.Tensor) or a.device.type != 'npu':
        raise TypeError("tensor should be torch.Tensor, and device type should be npu" + pta_error(ErrCode.PARAM))
    if not isinstance(b, torch.Tensor) or b.device.type != 'npu':
        raise TypeError("tensor should be torch.Tensor, and device type should be npu" + pta_error(ErrCode.PARAM))
    if not isinstance(c, torch.Tensor) or c.device.type != 'npu':
        raise TypeError("tensor should be torch.Tensor, and device type should be npu" + pta_error(ErrCode.PARAM))
    if (a.dtype not in _ABFT_SUPPORTED_DTYPES or b.dtype not in _ABFT_SUPPORTED_DTYPES
            or c.dtype not in _ABFT_SUPPORTED_DTYPES):
        raise TypeError(f"matmul_checksum only supports bfloat16 and float32, but got "
                        f"a.dtype={a.dtype}, b.dtype={b.dtype}, c.dtype={c.dtype}" + pta_error(ErrCode.PARAM))


def _matmul_checksum_py(a, b, c):
    r"""Legacy pure-PyTorch checksum, used as fallback when the op is unusable.

    Args:
        a(Tensor): matmul's input parameter a, and the device must be npu.
        b(Tensor): matmul's input parameter b, and the device must be npu.
        c(Tensor): matmul's output result c, and the device must be npu.

    Returns: The bool scalar tensor, located on the npu side, indicates whether there are any anomalies in the calculation result.
    """
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


def _matmul_checksum(a, b, c):
    r"""Compare whether there are any feature anomalies in the calculation results of matmul.

    Dispatches to the V-ABFT op ``torch.ops.npu._npu_matmul_abft_verify`` when
    it is available and the inputs fit its contract (2D bf16/fp32 with
    consistent shapes); otherwise falls back to the pure-PyTorch
    implementation. Input validation is shared by both paths.

    Args:
        a(Tensor): matmul's input parameter a, and the device must be npu.
        b(Tensor): matmul's input parameter b, and the device must be npu.
        c(Tensor): matmul's output result c, and the device must be npu.

    Returns: The bool scalar tensor, located on the npu side, indicates whether there are any anomalies in the calculation result.
    """
    _check_matmul_checksum_inputs(a, b, c)
    if _abft_path_applicable(a, b, c) and _npu_matmul_abft_verify_available():
        return _matmul_checksum_abft(a, b, c)
    return _matmul_checksum_py(a, b, c)
