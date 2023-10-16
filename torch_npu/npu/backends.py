# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
from typing import Union

import torch
import torch_npu._C


def preferred_linalg_library(backend: Union[None, str, torch._C._LinalgBackend] = None) -> torch._C._LinalgBackend:
    """Currently, the linalg lib is not available and does not support it.
    By default, it returns Default type.
    """
    warnings.warn("torch.npu.preferred_linalg_library isn't implemented!")
    return torch._C._LinalgBackend.Default
