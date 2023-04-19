# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import builtins
from builtins import isinstance as builtin_isinstance

import torch
from torch._dynamo import eval_frame, allowed_functions


class NpuNullBackendCtx:

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        self.prior_isinstance = builtins.isinstance
        self.prior_tensor_device = torch.Tensor.device
        self.prior_device = torch.device
        builtins.isinstance = builtin_isinstance
        torch.Tensor.device = torch._C._TensorBase.device
        torch.device = torch._C.device
        allowed_functions._builtin_function_ids.add(id(builtin_isinstance))
        return self.enter_result

    def __exit__(self, *excinfo):
        builtins.isinstance = self.prior_isinstance
        torch.Tensor.device = self.prior_tensor_device
        torch.device = self.prior_device


def add_dynamo_patch():
    eval_frame.null_context = NpuNullBackendCtx
