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

from typing import Optional, List

import torch
from torch.onnx import symbolic_helper

import torch_npu


class wrapper_npu_transpose(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_transpose(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, perm: List[int], require_contiguous: bool = True):
        return g.op("npu::NPUTranspose", self, perms_i=perm,
                    require_contiguous_i=require_contiguous)


class wrapper_npu_broadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_broadcast(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, size: List[int]):
        return g.op("npu::NPUBroadcast", self, sizes_i=size)


class wrapper_npu_one_hot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_one_hot(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, num_classses: int = -1, depth: int = 1,
                 on_value: int = 1, off_value: int = 0):
        return g.op("npu::NPUOneHot", self, num_classses_i=num_classses, depth_i=depth,
                    on_value_i=on_value, off_value_i=off_value)


class wrapper_npu_slice(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_slice(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, offsets: List[int], size: List[int]):
        return g.op("npu::NPUSlice", self, offsetss_i=offsets, sizes_i=size)


def torch_wrapper_npu_transpose(self: torch.Tensor, perm: List[int],
                                require_contiguous: bool = True):
    return wrapper_npu_transpose.apply(self, perm, require_contiguous)


def torch_wrapper_npu_broadcast(self: torch.Tensor, size: List[int]):
    return wrapper_npu_broadcast.apply(self, size)


def torch_wrapper_npu_one_hot(self: torch.Tensor, num_classses: int = -1, depth: int = 1,
                              on_value: int = 1, off_value: int = 0):
    return wrapper_npu_one_hot.apply(self, num_classses, depth, on_value, off_value)


def torch_wrapper_npu_slice(self: torch.Tensor, offsets: List[int], size: List[int]):
    return wrapper_npu_slice.apply(self, offsets, size)


def add_onnx_ops():
    torch_npu.npu_transpose = torch_wrapper_npu_transpose
    torch_npu.npu_broadcast = torch_wrapper_npu_broadcast
    torch_npu.npu_one_hot = torch_wrapper_npu_one_hot
    torch_npu.npu_slice = torch_wrapper_npu_slice
