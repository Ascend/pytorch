__all__ = ["autocast", "custom_fwd", "custom_bwd"]


import functools
import collections
from typing import Any
from typing_extensions import deprecated

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

import torch
import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.npu.amp.autocast(args...)`` is equivalent to ``torch.autocast("npu", args...)``
    """

    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16, cache_enabled: bool = True):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "npu"
            self.fast_dtype = dtype
            return
        super().__init__("npu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (value.is_floating_point() and value.device.type == 'npu' and (value.dtype is not torch.float64))
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = map(lambda v: _cast(v, dtype), value)
        if isinstance(value, list) or isinstance(value, tuple):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


@deprecated(
    "`torch_npu.npu.amp.custom_fwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_fwd(args..., device_type='npu')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch_npu.npu.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='npu')`` instead.
    """
    return functools.partial(torch.amp.custom_fwd, device_type="npu")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`torch_npu.npu.amp.custom_bwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_bwd(args..., device_type='npu')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``torch_npu.npu.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='npu')`` instead.
    """
    return functools.partial(torch.amp.custom_bwd, device_type="npu")(bwd)