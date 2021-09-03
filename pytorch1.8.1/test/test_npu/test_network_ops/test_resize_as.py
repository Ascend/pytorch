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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestResizeAs(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.resize_as_(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.resize_as_(input1, input2)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_resize_as_type_format(self, device):
        shape_format = [
                [torch.float32, (1, 2), (3, 4)],
                [torch.float32, (1, 2, 5), (3, 4, 7)],
                [torch.float16, (2, 3, 4), (5, 6, 7)]
        ]

        for item in shape_format:
            cpu_input1 = torch.randn(item[1])
            cpu_input2 = torch.randn(item[2])

            if item[0] == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float16)
                cpu_input2 = cpu_input2.to(torch.float16)

            npu_input1 = cpu_input1.npu()
            npu_input2 = cpu_input2.npu()

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)

            self.assertEqual(cpu_output.shape, npu_output.shape)


instantiate_device_type_tests(TestResizeAs, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

