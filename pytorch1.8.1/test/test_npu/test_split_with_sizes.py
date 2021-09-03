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


class Test_split_with_sizes(TestCase):
    def cpu_op_exec(self, input1, split_sizes, dim):
        outputs = torch.split_with_sizes(input1, split_sizes, dim)
        outputs_np = []
        for output in outputs:
            outputs_np.append(output.numpy())
        return outputs_np

    def npu_op_exec(self, input1, split_sizes, dim):
        input1 = input1.to("npu")
        outputs = torch.split_with_sizes(input1, split_sizes, dim)
        outputs = list(outputs)
        output_cpu = []
        output_np = []
        for i in outputs:
            output_cpu.append(i.to("cpu"))
        for i in output_cpu:
            output_np.append(i.numpy())
        return output_np

    def test_add_common_shape_format1(self, device):
        shape_format = [  # input, split_sizes, dim
                [[np.int32, -1, (2, 3)], [1, 1], 0],
                [[np.int32, -1, (2, 3)], [1, 1, 1], 1],
                [[np.int32, -1, (2, 3, 10)], [2, 3, 5], 2],
                [[np.int32, -1, (2, 3, 10, 4, 5)], [1, 1, 1, 1], 3],
                [[np.int32, -1, (2, 3, 10, 4, 5)], [1, 1, 1, 1, 1], 4]
                ]
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)

            split_sizes = item[1]
            dim = item[2]
            cpu_outputs = self.cpu_op_exec(cpu_input1, split_sizes, dim)
            npu_outputs = self.npu_op_exec(npu_input1, split_sizes, dim)
            for i in range(0, len(cpu_outputs)):
                self.assertRtolEqual(cpu_outputs[i], npu_outputs[i])

    def test_add_common_shape_format2(self, device):
        shape_format = [  # input, split_sizes, dim
                [[np.float32, -1, (10, 31, 149, 2)], [2, 3, 5], 0],
                [[np.float32, -1, (10, 31, 149, 2)], [2, 3, 5, 10, 11], 1],
                [[np.float32, -1, (10, 31, 149, 2)], [50, 50, 20, 29], 2],
                [[np.float32, -1, (10, 31, 149, 2)], [25, 25, 25, 25, 20, 29], 2],
                [[np.float32, -1, (10, 31, 149, 2)], [1, 1], 3]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -1.1754943508e-38, -1.1754943508e-38)
            split_sizes = item[1]
            dim = item[2]
            cpu_outputs = self.cpu_op_exec(cpu_input1, split_sizes, dim)
            npu_outputs = self.npu_op_exec(npu_input1, split_sizes, dim)
            for i in range(0, len(cpu_outputs)):
                self.assertRtolEqual(cpu_outputs[i], npu_outputs[i])


instantiate_device_type_tests(Test_split_with_sizes, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
