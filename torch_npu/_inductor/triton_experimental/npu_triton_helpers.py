# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Copyright (c) 2013 the respective contributors
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
from triton.language import core
from torch._inductor.runtime import triton_helpers

try:
    libdevice = tl.extra.cann.libdevice
except AttributeError:
    libdevice = tl.extra.ascend.libdevice
math = tl.math


@triton.jit
def minimum(a, b):
    mask = a < b
    if a.dtype.is_floating():
        mask |= a != a
    return tl.where(mask, a, b)


triton_helpers.minimum = minimum


@triton.jit
def maximum(a, b):
    if a.dtype.is_bool():
        return a | b
    mask = a > b
    if a.dtype.is_floating():
        mask |= a != a
    return tl.where(mask, a, b)


triton_helpers.maximum = maximum


@triton.jit
def minimum(a, b):
    if a.dtype.is_bool():
        return a & b
    mask = a < b
    if a.dtype.is_floating():
        mask |= a != a
    return tl.where(mask, a, b)


triton_helpers.minimum = minimum


@triton.jit
def max2(a, dim):
    return tl.max(a, dim, propagate_nan=True)


@triton.jit
def _min_prop_nan(a, b):
    return core.minimum(a, b, propagate_nan=core.PropagateNan.ALL)


@triton.jit
def min2(a, dim):
    return tl.reduce(a, dim, _min_prop_nan)


triton_helpers.max2 = max2
triton_helpers.min2 = min2


@triton.jit
def lt(a, b):
    if a.dtype.is_bool():
        return b & (a != b)
    return a < b


@triton.jit
def gt(a, b):
    if a.dtype.is_bool():
        return a & (a != b)
    return a > b


@triton.jit
def le(a, b):
    if a.dtype.is_bool():
        return b | (a == b)
    return a <= b


@triton.jit
def ge(a, b):
    if a.dtype.is_bool():
        return a | (a == b)
    return a >= b


@triton.jit
def log1p(x):
    """Accurate ``log(1 + x)`` for the vector path.

    CANN's ``libdevice.log1p`` is implemented as ``log(1 + x)``: the addition
    is rounded to the input dtype before the log is applied, so inputs below
    half an ulp of 1 collapse to 0 and the next decades lose most of their
    bits (fp32: 1e-8 -> 0, 1e-7 -> 19% high). ``u = 1 + x`` keeps only the leading bits of x;
    ``delta = x - (u - 1)`` is exactly the part the addition dropped and
    ``log(u) + delta / u`` restores ``log1p`` to about 1 ulp.

    ``x = -1`` gives ``u = 0``; the divisor is forced away from 0 so that
    ``log(0) + 0 / 0`` stays ``-inf`` instead of becoming NaN. ``x = +inf``
    gives ``u = +inf`` and ``delta = inf - inf = NaN``; the correction is
    dropped for non-finite ``u`` so the result stays ``+inf`` (``-inf`` and
    NaN still produce NaN).

    ``x = -0.0`` needs one more fix-up: ``log1p(-0.0)`` is ``-0.0``, but the
    compensated form evaluates ``log(1) + 0 - 0``, and every term of that is
    ``+0``, so the sign of the input zero is lost. That changes
    ``1 / log1p(-0.0)`` from ``-inf`` to ``+inf``. Return the argument itself
    for zero inputs (it already carries the sign) and the compensated result
    everywhere else.
    """
    u = 1.0 + x
    delta = x - (u - 1.0)
    # u is non-finite for x = +-inf/NaN. log(u) already carries the correct
    # +inf/NaN then, while delta would poison it with NaN (inf - inf).
    # u - u == 0 holds for finite u only.
    delta = tl.where(u - u == 0.0, delta, 0.0)
    result = tl.log(u) + delta / tl.where(u == 0.0, 1.0, u)
    return tl.where(x == 0.0, x, result)
