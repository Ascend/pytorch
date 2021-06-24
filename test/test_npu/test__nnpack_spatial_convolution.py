# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import unittest

class TestNnpackSpatialConvolution(TestCase):

    def generate_data(self, min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype):
        input_shape = (N, C0, Hi, Wi)
        input_x = np.random.uniform(min_d, max_d, input_shape).astype(dtype)
        weight_shape = (C1, C0, Hw, Ww)
        weight = np.random.uniform(min_d, max_d, weight_shape).astype(dtype)
        input_x = torch.from_numpy(input_x)
        weight = torch.from_numpy(weight)
        bias = np.zeros(C1).astype(dtype)
        bias = torch.from_numpy(bias)
        padding = tuple(np.ones(2).astype(np.int))
        return input_x, weight, bias, padding 

    @unittest.skipIf(not torch._nnpack_available(),"NNPACK unavailable")
    def cpu_op_exec(self, input_x, weight, bias, padding):
        flag = 0
        if input_x.dtype == torch.float16:
            input_x = input_x.to(torch.float32)
            weight = weight.to(torch.float32)
            bias = bias.to(torch.float32)
            flag = 1
        output = torch._nnpack_spatial_convolution(
            input_x, weight, bias, padding)
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy()
        return output
        
    @unittest.skipIf(not torch._nnpack_available(),"NNPACK unavailable")
    def npu_op_exec(self, input_x, weight, bias, padding):
        flag = 0
        if input_x.dtype == torch.float16:
            input_x = input_x.to(torch.float32)
            weight = weight.to(torch.float32)
            bias = bias.to(torch.float32)
            flag = 1
        input_x = input_x.to("npu")
        weight = weight.to("npu")
        bias = bias.to("npu")
        output = torch._nnpack_spatial_convolution(
            input_x, weight, bias, padding)
        output = output.to("cpu")
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy()
        return output
        

    def test__nnpack_spatial_convolution_float16_1(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -2, 2, 1, 3, 4, 4, 2, 2, 2, np.float16)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float16_2(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -50, 50, 1, 3, 5, 5, 5, 2, 2, np.float16)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float16_3(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -50, 50, 1, 5, 1024, 1024, 5, 8, 8, np.float16)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float16_4(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -100, 100, 1, 5, 1024, 1024, 5, 8, 8, np.float16)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)
    

    def test__nnpack_spatial_convolution_float32_1(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -2, 2, 1, 3, 4, 4, 2, 2, 2, np.float32)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float32_2(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -50, 50, 1, 3, 4, 4, 2, 2, 2, np.float32)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float32_3(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -50, 50, 1, 5, 512, 512, 5, 8, 8, np.float32)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test__nnpack_spatial_convolution_float32_4(self, device):
        input_x, weight, bias, padding = self.generate_data( 
                #min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype
                 -100, 100, 1, 5, 512, 512, 5, 8, 8, np.float32)
        cpu_output = self.cpu_op_exec(input_x, weight, bias, padding)
        npu_output = self.npu_op_exec(input_x, weight, bias, padding)
        self.assertRtolEqual(cpu_output, npu_output)
    

instantiate_device_type_tests(TestNnpackSpatialConvolution, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()


