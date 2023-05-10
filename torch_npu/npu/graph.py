# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Huawei Technologies Co., Ltd
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

import torch_npu
from .utils import _lazy_init


def enable_graph_mode(verbose=False):
    torch_npu._C._npu_enable_graph_mode(verbose)


def disable_graph_mode():
    _lazy_init()
    torch_npu._C._npu_disable_graph_mode()


def is_graph_mode() -> bool:
    return torch_npu._C._npu_is_graph_mode()


def launch_graph():
    _lazy_init()
    if not is_graph_mode():
        raise RuntimeError("Npu run mode must be graph mode when launch graph")
    torch_npu._C._npu_launch_graph()