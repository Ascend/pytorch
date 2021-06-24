# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import torch 
import numpy as np 
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestAvgPool1d(TestCase):
    def generate_bool(self):
        scalar = random.randint(1, 2)
        return scalar == 1

    def generate_int(self, min_d, max_d):
        scalar = random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, size, strd, pad, mode, include_pad):
        output = torch.nn.functional.avg_pool1d(input1, kernel_size=size, stride=strd, padding=pad, ceil_mode=mode, count_include_pad=include_pad)
        return output 
    
    def npu_op_exec(self, input1, size, strd, pad, mode, include_pad): 
        output = torch.nn.functional.avg_pool1d(input1, kernel_size=size, stride=strd, padding=pad, ceil_mode=mode, count_include_pad=include_pad)
        output = output.to("cpu")
        return output   

    def test_avg_pool1d_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (3, 3, 3)]],
            [[np.float32, -1, (4, 4, 4)]],
            [[np.float32, 0, (3, 3, 3)]],
            [[np.float32, 3, (3, 3, 3)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_size = self.generate_int(1, 3)
            cpu_strd = self.generate_int(4, 5)
            cpu_pad = self.generate_int(0, 5)
            cpu_mode = self.generate_bool()
            cpu_include_pad = self.generate_bool()
            npu_size = cpu_size
            npu_strd = cpu_strd
            npu_pad = cpu_pad
            npu_mode = cpu_mode
            npu_include_pad = cpu_include_pad
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_size, cpu_strd, cpu_pad, cpu_mode, cpu_include_pad)
            npu_output = self.npu_op_exec(npu_input1, npu_size, npu_strd, npu_pad, npu_mode, npu_include_pad)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())

    def test_avg_pool1d_float16_shape_format(self, device):
        shape_format = [
            [[np.float16, -1, (4, 5, 5)]],
            [[np.float16, -1, (5, 5, 4)]],
            [[np.float16, 0, (4, 5, 5)]],
            [[np.float16, 3, (4, 5, 5)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            size = self.generate_int(1, 3)
            strd = self.generate_int(4, 5)
            pad = self.generate_int(0, 5)
            mode = self.generate_bool()
            include_pad = self.generate_bool()
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, size, strd, pad, mode, include_pad)
            npu_output = self.npu_op_exec(npu_input1, size, strd, pad, mode, include_pad)
            self.assertRtolEqual(cpu_output.numpy().astype(np.float16), npu_output.numpy().astype(np.float16))
     
instantiate_device_type_tests(TestAvgPool1d, globals(), except_for='cpu')
if __name__ == '__main__': 
    torch.npu.set_device("npu:5")
    run_tests() 
