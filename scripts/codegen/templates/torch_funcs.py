# Copyright (c) 2020 Huawei Technologies Co., Ltd
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


import torch
import torch_npu


def jit_script(obj, optimize=None, _frames_up=0, _rcb=None):
    # (Ascend) Disable extension of torch.jit.script
    return obj

def add_torch_funcs():
    torch.tensor = torch_npu.tensor
    torch.full = torch_npu.full
    torch.randint = torch_npu.randint
    torch.range = torch_npu.range
    torch.arange = torch_npu.arange
    torch.empty_with_format = torch_npu.empty_with_format
    torch.jit.script = jit_script

${device_methods_def_py}