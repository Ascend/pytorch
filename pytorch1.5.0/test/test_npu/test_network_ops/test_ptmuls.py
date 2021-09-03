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

class TestPtMuls(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.mul(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.mul(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_ptmuls_shape_format_fp16(self, device):
        format_list = [0, 3, 4, 29]
        shape_list = [(64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2 = 4.0
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, cpu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ptmuls_shape_format_fp32(self, device):
        format_list = [0, 3, 4, 29]
        shape_list = [(64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2 = 6.2
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, cpu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_output1 = self.cpu_op_exec(cpu_input2, cpu_input1)
            npu_output1 = self.npu_op_exec(cpu_input2, npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)

instantiate_device_type_tests(TestPtMuls, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

