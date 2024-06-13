r"""Functional interface"""

__all__ = ["dropout_with_byte_mask"]

import torch
import torch_npu as _VF
from torch_npu.utils._error_code import ErrCode, ops_error

Tensor = torch.Tensor


def dropout_with_byte_mask(input1, p=0.5, training=True, inplace=False):
    # type: (Tensor, float, bool, bool) -> Tensor
    r"""Applies an NPU compatible dropout_with_byte_mask operation, Only supports npu devices. 

    This dropout_with_byte_mask method generates stateless random uint8 mask and do dropout according to the mask.

    .. note::
        The performance is improved only in the device 32 core scenario.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0. or p > 1.:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p) + ops_error(ErrCode.VALUE))
    if inplace:
        raise ValueError("dropout probability has no-inplace computing." + ops_error(ErrCode.VALUE))
    return (_VF.dropout_with_byte_mask_(input1, p, training)
            if inplace
            else _VF.dropout_with_byte_mask(input1, p, training))
