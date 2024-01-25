# Copyright (c) 2021 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import functools
import collections
from typing import Any, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

from torch.types import _dtype

import torch
from torch._six import string_classes
from .common import amp_definitely_not_available


class npu_autocast(torch.autocast_mode.autocast):
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, NPU ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``.half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example::

        # Creates model and optimizer in default precision
        model = Net().npu()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @autocast()
            def forward(self, input):
                ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the autocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.  Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="npu")
        b_float32 = torch.rand((8, 8), device="npu")
        c_float32 = torch.rand((8, 8), device="npu")
        d_float32 = torch.rand((8, 8), device="npu")

        with autocast():
            # torch.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = torch.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_float16.float())

    CPU Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cpu")
        b_float32 = torch.rand((8, 8), device="cpu")
        c_float32 = torch.rand((8, 8), device="cpu")
        d_float32 = torch.rand((8, 8), device="cpu")

        with autocast(dtype=torch.bfloat16, device_type="cpu"):
            # torch.mm is on autocast's list of ops that should run in bfloat16.
            # Inputs are float32, but the op runs in bfloat16 and produces bfloat16 output.
            # No manual casts are required.
            e_bfloat16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_bfloat16 = torch.mm(d_float32, e_bfloat16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_bfloat16.float())

    Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling autocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="npu")
        b_float32 = torch.rand((8, 8), device="npu")
        c_float32 = torch.rand((8, 8), device="npu")
        d_float32 = torch.rand((8, 8), device="npu")

        with autocast():
            e_float16 = torch.mm(a_float32, b_float32)

            with autocast(enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = torch.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the autocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = torch.mm(d_float32, f_float32)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one NPU per process
    (see :ref:`Working with Multiple NPUs<amp-multinpu>`).

    Args:
        device_type(string, required):  Whether to use 'npu' or 'cpu' device
        enabled(bool, optional, default=True):  Whether autocasting should be enabled in the region.
        dtype(torch_dtype, optional):  Whether to use torch.float16 or torch.bfloat16.
        cache_enabled(bool, optional, default=True):  Whether the weight cache inside autocast should be enabled.
    """

    def __init__(self, device_type: str,
                 dtype: Optional[_dtype] = None,
                 enabled: bool = True,
                 cache_enabled: Optional[bool] = None):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            self.fast_dtype = dtype
            if dtype is None:
                raise ValueError("dtype is None")
            return
        self.device = device_type
        if self.device == 'npu':
            self.fast_dtype = torch.get_autocast_gpu_dtype()
        elif self.device == 'cpu':
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        else:
            raise RuntimeError('User specified autocast device_type must be \'npu\' or \'cpu\'')
        self._cache_enabled = torch.is_autocast_cache_enabled()
        if amp_definitely_not_available() and self.device == 'npu':
            warnings.warn('User provided device_type of \'npu\', but NPU is not available. Disabling')
            enabled = False
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled

        if self.device == 'cpu':
            supported_dtype = [torch.bfloat16]
            if self.fast_dtype not in supported_dtype:
                error_message = 'In CPU autocast, but the target dtype is not supported. Disabling autocast.\n'
                error_message += 'CPU Autocast only supports dtype of torch.bfloat16 currently.'
                warnings.warn(error_message)
                enabled = False
        if self.device == 'npu':
            if self.fast_dtype == torch.bfloat16 and not torch.npu.is_bf16_supported():
                raise RuntimeError('Current NPU Device does not support bfloat16. Please switch dtype to float16.')
        self._enabled = enabled

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        if torch._jit_internal.is_scripting():
            return None
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


class autocast(npu_autocast):
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        if torch._jit_internal.is_scripting():
            return
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
    elif isinstance(value, string_classes):
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


# custom_fwd is a decorator that may or may not be used with arguments, following
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
                raise ValueError("More than one keyword argument.")
            cast_inputs = kwargs.get("cast_inputs", None)
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    if len(kwargs) == 0:
        cast_inputs = None
    else:
        if len(kwargs) != 1:
            raise ValueError("More than one keyword argument.")
        cast_inputs = kwargs.get("cast_inputs", None)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled()
            return fwd(*args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled()
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
        with autocast(args[0]._fwd_used_autocast):
            return bwd(*args, **kwargs)

    return decorate_bwd


def apply_autocast_patch():
    torch.autocast = npu_autocast
