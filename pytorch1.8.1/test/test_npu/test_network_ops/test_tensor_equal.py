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


class TestEqual(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.equal(input1, input2)
        output = np.array(output, dtype=np.int32)
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.equal(input1, input2)
        output = np.array(output, dtype=np.int32)
        return output

    def test_equal_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, -100, 100)

            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)

            cpu_output1 = self.cpu_op_exec(cpu_input1, cpu_input1)
            npu_output1 = self.npu_op_exec(npu_input1, npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)

            cpu_output0 = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output0 = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output0, npu_output0)

    def test_equal_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, -100, 100)

            cpu_output1 = self.cpu_op_exec(cpu_input1, cpu_input1)
            npu_output1 = self.npu_op_exec(npu_input1, npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)

            cpu_output0 = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output0 = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output0, npu_output0)


instantiate_device_type_tests(TestEqual, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
