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
from torch._inductor.runtime import triton_helpers

try:
    libdevice = tl.extra.cann.libdevice
except AttributeError:
    libdevice = tl.extra.ascend.libdevice
math = tl.math


@triton.jit
def max2(a, dim):
    return tl.max(a, dim, propagate_nan=True)


@triton.jit
def min2(a, dim):
    return tl.min(a, dim, propagate_nan=True)


triton_helpers.max2 = max2
triton_helpers.min2 = min2
