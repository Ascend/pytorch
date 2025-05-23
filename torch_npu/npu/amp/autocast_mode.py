import functools
import collections
from typing import Any

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


# custom_fwd is a decorator that may or may not be used with arguments.
# this works:
#     @custom_fwd
#     def forward(...):
# this also works:
#     @custom_fwd(cast_inputs=torch.float)
#     def forward(...):
# def custom_fwd(fwd=None, *, cast_inputs=None) with internal changes following the link above.
def custom_fwd(fwd=None, **kwargs):
    """
    Helper decorator for ``forward`` methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).  See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point NPU Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if fwd is None:
        if len(kwargs) == 0:
            cast_inputs = None
        else:
            if len(kwargs) != 1:
                raise ValueError("More than one keyword argument." + pta_error(ErrCode.PARAM))
            cast_inputs = kwargs.get("cast_inputs", None)
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    if len(kwargs) == 0:
        cast_inputs = None
    else:
        if len(kwargs) != 1:
            raise ValueError("More than one keyword argument." + pta_error(ErrCode.PARAM))
        cast_inputs = kwargs.get("cast_inputs", None)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = torch.get_autocast_dtype("npu")
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch_npu._C.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch_npu._C.is_autocast_enabled()
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(enabled=False):
                    return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
            else:
                return fwd(*args, **kwargs)

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd):
    """
    Helper decorator for backward methods of custom autograd functions (subclasses of
    :class:`torch.autograd.Function`).
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.
    """

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(args[0]._fwd_used_autocast, dtype=args[0]._dtype):
            return bwd(*args, **kwargs)

    return decorate_bwd
