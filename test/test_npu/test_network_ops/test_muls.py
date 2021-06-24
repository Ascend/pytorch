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

import sys
sys.path.append('..')
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestMuls(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.mul(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.mul(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def cpu_op_out_exec(self, input1, input2, input3):
        torch.mul(input1, input2, out = input3)
        input3 = input3.numpy()
        return input3

    def npu_op_out_exec(self, input1, input2, input3):
        torch.mul(input1, input2, out = input3)
        input3 = input3.to("cpu")
        input3 = input3.numpy()
        return input3

    def test_muls_shape_format_fp16(self, device):
        format_list = [0, 3, 4, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7), (2, 0, 2)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_muls_shape_format_fp32(self, device):
        format_list = [0, 3, 4, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_muls_shape_format_bool(self, device):
        format_list = [0]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.int32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1 > 50, cpu_input2 > 50)
            npu_output = self.npu_op_exec(npu_input1 > 50, npu_input2 > 50)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_muls_shape_format_out_fp32(self, device):
        format_list = [0]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpuout = torch.randn(6)
            npuout = torch.randn(6).to("npu")
            cpu_output = self.cpu_op_out_exec(cpu_input1, cpu_input2, cpuout)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, npuout)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_muls_mix_dtype(self, device):
        npu_input1, npu_input2 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input2, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_muls_scalar_dtype(self, device):
        cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(cpu_input1, 0.5)
        npu_output = self.npu_op_exec(npu_input1, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestMuls, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
