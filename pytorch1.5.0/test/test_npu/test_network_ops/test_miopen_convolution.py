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


class TestMiopenConvolution(TestCase):

    def op_exec_cpu(self, input, weight, bias, stride, padding, dilation, transposed, 
                    output_padding, groups, benchmark, deterministic, cudnn_enabled):

        cpuOutput = torch._convolution(input, weight, bias, stride, padding, dilation, transposed=False, output_padding=[0, 0], 
                                       groups=1, benchmark=False, deterministic=False, cudnn_enabled=False) 

        return cpuOutput

    def op_exec_npu(self, input, weight, bias, stride, padding, dilation, transposed, 
                    output_padding, groups, benchmark, deterministic, cudnn_enabled):

        input = input.to("npu")
        weight = weight.to("npu")
        bias = bias.to("npu")
        npuOutput = torch._convolution(input, weight, bias, stride, padding, dilation, transposed=False, output_padding=[0, 0],
                                      groups=1, benchmark=False, deterministic=False, cudnn_enabled=False)
        npuOutput = npuOutput.to("cpu")

        return npuOutput
    
    def test_miopen_convolution_float16_001(self, device):

        # input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, 
        # benchmark, deterministic, cudnn_enabled
        item = [[np.float16, 3, [2, 1, 5, 5]], [np.float16, 3, (1, 1, 1, 1)], [np.float16, 3, (1)], 
              [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, False]
        
        input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
        if input_cpu.dtype == torch.float16:
            input_cpu = input_cpu.to(torch.float32)
        weight_cpu, weight_npu = create_common_tensor(item[1], 0, 10)
        if weight_cpu.dtype == torch.float16:
            weight_cpu = weight_cpu.to(torch.float32)
        bias_cpu, bias_npu = create_common_tensor(item[2], 0, 10)
        if bias_cpu.dtype == torch.float16:
            bias_cpu = bias_cpu.to(torch.float32)
        
        cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, bias_cpu, stride=item[3], padding=item[4], dilation=item[5], transposed=item[6],
                                      output_padding=item[7], groups=item[8], benchmark=item[9], deterministic=item[10], cudnn_enabled=item[10])
        npu_output = self.op_exec_npu(input_npu, weight_npu, bias_npu, stride=item[3], padding=item[4], dilation=item[5], transposed=item[6],
                                      output_padding=item[7], groups=item[8], benchmark=item[9], deterministic=item[10], cudnn_enabled=item[10])
        cpu_output = cpu_output.to(npu_output.dtype)
                
        print("======cpuOutput_float16_001======")
        print(cpu_output)
        print("======npuOutput_float16_001======")
        print(npu_output)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
    

instantiate_device_type_tests(TestMiopenConvolution, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
