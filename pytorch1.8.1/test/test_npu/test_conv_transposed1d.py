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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestConv1DTranspose(TestCase):
    def cpu_op_exec(self, input, weight, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation):
        m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        m.weight.data = weight
        output = m(input)
        output = output.detach().numpy()
        return output
        
    def npu_op_exec(self, input, weight, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation):
        m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        m.weight.data = weight
        output = m(input)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output
        
    def test_conv_transpose1d_common_shape_format(self, device):
        shape_format = [ 
            [[np.float32, 3, (256, 8, 1, 1)], [np.float32, 3, (8, 32, 1, 1)], (1, 1), 0, 1, None, (1, 1)],
            [[np.float32, 3, [256, 32, 112, 112]], [np.float32, 0, [32, 16, 1, 1]], 1, 0, 1, None,  1],
            [[np.float32, 0, [256, 32, 224, 224]], [np.float32, 0, [32, 3, 3, 3]], [2, 2], 0, 1, None,  1],
            [[np.float32, 3, [1024, 116, 14, 14]], [np.float32, 4, [116, 116, 1, 1]],1, 0, 1, True, 1],
            [[np.float32, 0, [1024, 3, 224, 224]], [np.float32, 4, [3, 24, 3, 3]], [2, 2], 0, 1, None, 1],
            [[np.float32, 3, [1024, 24, 56, 56]], [np.float32, 4, [24, 24, 1, 1]],  1, 0, 1, True, 1],
        ]
            
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 10)
            cpu_weight, npu_weight = create_common_tensor(item[1], 1, 10)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.cpu_op_exec(cpu_input, cpu_weight, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          stride=item[2], padding=item[3], output_padding = item[3], groups=item[4], bias=item[5], dilation=item[6])
            npu_output = self.npu_op_exec(cpu_input, cpu_weight, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          stride=item[2], padding=item[3], output_padding = item[3], groups=item[4], bias=item[5], dilation=item[6])
            self.assertRtolEqual(cpu_output, npu_output) 
 
    def test_conv1d_shape_format_float16(self, device):
        def cpu_op_exec_fp16(input, weight, in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation):
            input = input.to(torch.float32) 
            weight = weight.to(torch.float32)           
            m = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
            m.weight.data = weight
            output = m(input)
            output = output.detach().numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [  
            [[np.float16, 3, (256, 8, 1, 1)], [np.float16, 3, (8, 32, 1, 1)], (1, 1), 0, 1, None, (1, 1)],
            [[np.float16, 3, [256, 32, 112, 112]], [np.float16, 0, [32, 16, 1, 1]], 1, 0, 1, None,  1],
            [[np.float16, 0, [256, 32, 224, 224]], [np.float16, 0, [32, 3, 3, 3]], [2, 2], 0, 1, None,  1],
            [[np.float16, 3, [1024, 116, 14, 14]], [np.float16, 4, [116, 116, 1, 1]], 1, 0, 1, True, 1],
            [[np.float16, 0, [1024, 3, 224, 224]], [np.float16, 4, [3, 24, 3, 3]], [2, 2], 0, 1, None, 1],
            [[np.float16, 3, [1024, 24, 56, 56]], [np.float16, 4, [24, 24, 1, 1]],  1,  0, 1, True, 1],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 10)
            cpu_weight, npu_weight = create_common_tensor(item[1], 1, 10)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = cpu_op_exec_fp16(cpu_input, cpu_weight, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          stride=item[2], padding=item[3], output_padding = item[3], groups=item[4], bias=item[5], dilation=item[6])
            npu_output = self.npu_op_exec(cpu_input, cpu_weight, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          stride=item[2], padding=item[3], output_padding = item[3], groups=item[4], bias=item[5], dilation=item[6])
            self.assertRtolEqual(cpu_output, npu_output)  
   
instantiate_device_type_tests(TestConv1DTranspose, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()