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

class TestLinspace(TestCase):
    def generate_scalar(self, dtype, min, max):
        if dtype == "float32":
            scalar = np.random.uniform(min, max)
        if dtype == "int32":
            scalar = np.random.randint(min, max)
        return scalar

    def cpu_op_exec(self,start, end, steps):
        output = torch.linspace(start, end, steps)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self,start, end, steps, output):
        torch.linspace(start, end, steps, out=output)
        output = output.numpy()
        return output

    def npu_op_exec(self, start, end, steps):
        output = torch.linspace(start, end, steps=steps, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self,start, end, steps, output):
        torch.linspace(start, end, steps=steps, out=output, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_linspace_common_shape_format(self, device):
        shape_format = [
                ["int32", 5],
                ["float32", 3],
                ["float32", 50],
        ]
        for item in shape_format:
            cpu_start = npu_start = self.generate_scalar(item[0], 0, 10)
            cpu_end = npu_end = self.generate_scalar(item[0], 70, 100)
            steps = item[1]
            cpu_output = self.cpu_op_exec(cpu_start, cpu_end, steps)
            npu_output = self.npu_op_exec(cpu_start, cpu_end, steps)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_linspace_out_common_shape_format(self, device):
        shape_format = [
                ["int32", 5, [np.float32, 0, (5)]],
                ["float32", 3, [np.float32, 0, (3)]],
        ]
        for item in shape_format:
            cpu_start = npu_start = self.generate_scalar(item[0], 0, 10)
            cpu_end = npu_end = self.generate_scalar(item[0], 20, 30)
            steps = item[1]
            cpu_input2, npu_input2 = create_common_tensor(item[2], 0, 10)
            cpu_output = self.cpu_op_exec_out(cpu_start, cpu_end, steps, cpu_input2)
            npu_output = self.npu_op_exec_out(npu_start, npu_end, steps, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestLinspace, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
