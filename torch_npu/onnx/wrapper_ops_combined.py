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
from torch import _VF

import torch_npu


class NPULinearOP(object):

    @staticmethod
    def forward(input_, weight, bias=None):
        if torch.onnx.is_in_onnx_export():
            return torch._C._nn.linear(input_, weight, bias)
        return torch_npu._C._VariableFunctionsClass.npu_linear(input_, weight, bias)


class NPUTransposeOP(object):

    @staticmethod
    def forward(self, perm, require_contiguous=True, out=None):
        if torch.onnx.is_in_onnx_export():
            if require_contiguous:
                out = self.permute(perm).contiguous()
            else:
                out = self.permute(perm)
            return out
        out = torch_npu._C._VariableFunctionsClass.npu_transpose(self, perm, require_contiguous)
        return out


class NPUBroadcastOP(object):

    @staticmethod
    def forward(self, size, out=None):
        if torch.onnx.is_in_onnx_export():
            out = self.expand(size)
            return out
        out = torch_npu._C._VariableFunctionsClass.npu_broadcast(self, size)
        return out


class NPUOneOP(object):

    @staticmethod
    def forward(self):
        if torch.onnx.is_in_onnx_export():
            self = torch.ones_like(self)
            return self
        return torch_npu._C._VariableFunctionsClass.one_(self)


class NPUConvTranspose2dOP(object):

    @staticmethod
    def forward(input_, weight, bias, padding, output_padding, stride, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            return torch.conv_transpose2d(input_, weight, bias, stride, padding,
                                          output_padding, groups, dilation)
        return torch_npu._C._VariableFunctionsClass.npu_conv_transpose2d(input_, weight, bias,
                                                                         padding, output_padding,
                                                                         stride, dilation, groups)


class NPUConv2dOP(object):

    @staticmethod
    def forward(input_, weight, bias, stride, padding, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            return torch.conv2d(input_, weight, bias, stride, padding, dilation, groups)
        return torch_npu._C._VariableFunctionsClass.npu_conv2d(input_, weight, bias, stride,
                                                               padding, dilation, groups)


class NPUConv3dOP(object):

    @staticmethod
    def forward(input_, weight, bias, stride, padding, dilation, groups):
        if torch.onnx.is_in_onnx_export():
            return torch.conv3d(input_, weight, bias, stride, padding, dilation, groups)
        return torch_npu._C._VariableFunctionsClass.npu_conv3d(input_, weight, bias, stride,
                                                               padding, dilation, groups)


def add_ops_combined_for_onnx():
    torch_npu.npu_linear = NPULinearOP.forward
    torch_npu.npu_transpose = NPUTransposeOP.forward
    torch_npu.npu_broadcast = NPUBroadcastOP.forward
    torch_npu.one_ = NPUOneOP.forward
    torch_npu.npu_conv_transpose2d = NPUConvTranspose2dOP.forward
    torch_npu.npu_conv2d = NPUConv2dOP.forward
    torch_npu.npu_conv3d = NPUConv3dOP.forward
