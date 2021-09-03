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
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestMkldnnConvolution(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input_data, weight, bais):
        input_npu = input_data.to('cpu')
        output = torch._convolution(input_npu, weight, bais, (1, 1), (0,0), (1, 1), False, (0, 0), 1, False, False, False)
        output = output.numpy()
        return output

    def npu_op_exec(self, input_data, weight, bais):
        weight = weight.to('npu')
        bais = bais.to('npu')
        input_npu = input_data.to('npu')
        output = torch._convolution(input_npu, weight, bais, (1, 1), (0,0), (1, 1), False, (0, 0), 1, False, False, False)
        output = output.cpu().numpy()
        return output    
        
    def test_mkldnn_convolution_float32_1(self, device):
        input_data= self.generate_data(0, 100, (100, 1, 20, 30), np.float32)
        input_data = input_data.to('cpu')
        weight = torch.ones((1, 1, 1, 1))
        bais = torch.zeros((1))
        cpu_output = self.cpu_op_exec(input_data, weight, bais)
        npu_output = self.npu_op_exec(input_data, weight, bais)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_mkldnn_convolution_float32_2(self, device):
        input_data= self.generate_data(0, 100, (20, 1, 100, 20), np.float32)
        input_data = input_data.to('cpu')
        weight = torch.ones((1, 1, 1, 1))*3
        bais = torch.zeros((1)) * 1
        cpu_output = self.cpu_op_exec(input_data, weight, bais)
        npu_output = self.npu_op_exec(input_data, weight, bais)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_mkldnn_convolution_float16(self, device):
        input_data= self.generate_data(0, 100, (20, 1, 50, 20), np.float16)
        input_data_cpu = input_data.to(torch.float32)
        weight = torch.ones((1, 1, 1, 1)) *2
        bais = torch.zeros((1)) * 5
        weight_npu = weight.to(torch.float16)
        bais_npu = bais.to(torch.float16)
        cpu_output = self.cpu_op_exec(input_data_cpu, weight, bais).astype(np.float16)
        npu_output = self.npu_op_exec(input_data, weight_npu, bais_npu)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_mkldnn_convolution_float16_1(self, device):
        input_data= self.generate_data(0, 100, (2, 1, 5, 5), np.float16)
        input_data_cpu = input_data.to(torch.float32)
        weight = torch.ones((1, 1, 1, 1)) *2.3
        bais = torch.zeros((1)) * 0.5
        weight_npu = weight.to(torch.float16)
        bais_npu = bais.to(torch.float16)
        cpu_output = self.cpu_op_exec(input_data_cpu, weight, bais).astype(np.float16)
        npu_output = self.npu_op_exec(input_data, weight_npu, bais_npu)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_mkldnn_convolution_float32_3(self, device):
        input_data= self.generate_data(-100, 100, (60, 1, 100, 60), np.float32)
        input_data = input_data.to('cpu')
        weight = torch.ones((1, 1, 1, 1))*40.0
        bais = torch.ones((1)) * 5.0
        cpu_output = self.cpu_op_exec(input_data, weight, bais)
        npu_output = self.npu_op_exec(input_data, weight, bais)
        self.assertRtolEqual(cpu_output, npu_output)
        
instantiate_device_type_tests(TestMkldnnConvolution, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()
