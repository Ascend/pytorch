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
        self.torch_func_names = [
            "tensor", "full", "randint", "range", "arange", "as_tensor",
            "empty", "empty_like", "empty_strided", "eye", "ones", "ones_like",
            "rand", "rand_like", "randint_like", "randn", "randn_like",
            "randperm", "scalar_tensor", "zeros", "zeros_like"
        ]
        self.tensor_method_names = [
            "device", "new_full", "new_zeros", "new_ones", "new_tensor"
        ]

    def __enter__(self):
        # pylint: disable=expression-not-assigned
        # pylint: disable=attribute-defined-outside-init
        self.prior_isinstance = builtins.isinstance
        self.prior_tensor_device = torch.Tensor.device
        self.prior_device = torch.device
        self.prior_torch_funcs = [[name, getattr(torch, name)]
                                  for name in self.torch_func_names]
        self.prior_tensor_methods = [[name, getattr(torch.Tensor, name)]
                                     for name in self.tensor_method_names]
        builtins.isinstance = builtin_isinstance
        torch.device = torch._C.device
        [setattr(torch,name,getattr(torch._C._VariableFunctions, name))
         for name in self.torch_func_names]
        [setattr(torch.Tensor,name,getattr(torch._C._TensorBase, name))
         for name in self.tensor_method_names]
        allowed_functions._builtin_function_ids.add(id(builtin_isinstance))
        return self.enter_result

    def __exit__(self, *excinfo):
        # pylint: disable=expression-not-assigned
        builtins.isinstance = self.prior_isinstance
        torch.device = self.prior_device
        [setattr(torch, x[0], x[1]) for x in self.prior_torch_funcs]
        [setattr(torch.Tensor, x[0], x[1]) for x in self.prior_tensor_methods]


def add_dynamo_patch():
    eval_frame.null_context = NpuNullBackendCtx
