# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import torch.nn as nn
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
#pylint: disable=unused-argument

class TestMkldnnAdaptiveAvgPool2d(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, output_size):
        m = nn.AdaptiveAvgPool2d(output_size)
        output= m(input1) 
        return output.numpy()   

    def npu_op_exec(self, input1, output_size):
        m = nn.AdaptiveAvgPool2d(output_size).npu()
        output = m(input1)
        return output.cpu().numpy()

    def test_mkldnn_adaptiveAvgPool2d_shape_format_fp32(self, device):
        shape_list = [(32, 16, 16),
                      (16, 1024, 256),
                      (1024, 464, 11, 9),
                      (1, 2048, 15, 15)]
        output_list = [(4, 4), (3, 5), (1), (1, None), (None, 2),(2,1)]
        for item in shape_list:
            input1= self.generate_data(0, 100, item, np.float32)
            cpu_input1 = copy.deepcopy(input1)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input1, output_size)
                npu_output = self.npu_op_exec(input1, output_size)
                self.assertRtolEqual(cpu_output, npu_output)
      
    def test_mkldnn_adaptiveAvgPool2d_shape_format_fp16(self, device):
        def cpu_op_exec_fp16(input1, output_size):
            input1 = input1.to(torch.float32)
            m = nn.AdaptiveAvgPool2d(output_size)
            output= m(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
            
        def npu_op_exec_fp16(input1, output_size):
            input1 = input1.to(torch.float32)
            m = nn.AdaptiveAvgPool2d(output_size).npu()
            output = m(input1)
            output = output.to("cpu")
            output = output.numpy().astype(np.float16)
            return output 

        npu_input1 = self.generate_data(0, 100, (5,3,4), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = cpu_op_exec_fp16(cpu_input1, (4, 4))
        npu_output = npu_op_exec_fp16(npu_input1, (4, 4))
        self.assertRtolEqual(cpu_output, npu_output)  

instantiate_device_type_tests(TestMkldnnAdaptiveAvgPool2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()