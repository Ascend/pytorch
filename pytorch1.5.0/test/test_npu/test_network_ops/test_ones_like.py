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


class TestOnesLike(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.ones_like(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.ones_like(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_op_shape_format_fp16(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (1000, 1280), (32, 3, 3), (32, 144, 1, 1)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            print(item)
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_op_shape_format_fp32(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (1000, 1280), (32, 3, 3), (32, 144, 1, 1)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            print(item)
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestOnesLike, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
