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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestAddCMul(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def generate_output_data(self, min_d, max_d, shape, dtype):
        output_y = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_output_y = torch.from_numpy(output_y)
        return npu_output_y

    def cpu_op_exec(self, input1, input2, input3, scalar):
        output = torch.addcmul(input1, input2, input3, value=scalar)
        output = output.numpy()
        return output
    
    def cpu_op_exec_out(self, input1, input2, input3, scalar, output_y):
        output = output_y
        torch.addcmul(input1, input2, input3, value = scalar, out = output_y)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addcmul(input1, input2, input3, value=scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar, output_y):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = output_y.to("npu")
        torch.addcmul(input1, input2, input3, value=scalar, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_addcmul_3_3_float32(self, device):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_10_float32(self, device):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_addcmul_3_3_float16(self, device):
        input1, input2, input3 = self.generate_data(0, 100, (3, 3), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        cpu_output = self.cpu_op_exec(input1_cpu, input2_cpu, input3_cpu, 0.5).astype(np.float16)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_10_float16(self, device):
        input1, input2, input3 = self.generate_data(0, 100, (10, 10), np.float16)
        input1_cpu = input1.float()
        input2_cpu = input2.float()
        input3_cpu = input3.float()
        cpu_output = self.cpu_op_exec(input1_cpu, input2_cpu, input3_cpu, 0.5).astype(np.float16)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcmul_10_23_float32(self, device):
        input1, input2, input3 = self.generate_data(0, 100, (10, 23), np.float32)
        cpu_output = self.cpu_op_exec(input1, input2, input3, 0.5)
        npu_output = self.npu_op_exec(input1, input2, input3, 0.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def addcmul_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], -100, 100)
            cpu_input4, npu_input4 = create_common_tensor(item[1], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            if cpu_input3.dtype == torch.float16:
                cpu_input3 = cpu_input3.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, 2.0)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, 2.0, npu_input4)
            cpu_output = cpu_output.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_addcmul_out_result(self, device):
        shape_format = [
            [[np.float16, 0, [2, 6, 6, 3]], [np.float16, 0, [256, 116]]],
            [[np.float16, 0, [2, 3, 6]],  [np.float16, 0, [58, 58, 1, 1]]],
            [[np.float16, 0, [12, 2]],   [np.float16, 0, [128, 128]]],
            [[np.float16, 0, [12]], [np.float16, 0, [128, 116]]],
            [[np.float32, 0, [128, 64, 64, 32]], [np.float32, 0, [256, 116]]],
            [[np.float32, 0, [128, 32, 64]],   [np.float32, 0, [58, 58, 1, 1]]],
            [[np.float32, 0, [128, 32]],   [np.float32, 0, [128, 128]]],
            [[np.float32, 0, [128]], [np.float32, 0, [128, 116]]],
        ]
        self.addcmul_out_result(shape_format)


instantiate_device_type_tests(TestAddCMul, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
