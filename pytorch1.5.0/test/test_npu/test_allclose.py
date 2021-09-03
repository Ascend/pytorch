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

def create_all_one_tensor(item):
    dtype = item[0]
    format = item[1]
    shape = item[2]
    input1 = np.ones(shape).astype(dtype)
    cpu_input = torch.from_numpy(input1)
    npu_input = torch.from_numpy(input1).to("npu")
    if format != -1:
        npu_input = npu_input.npu_format_cast(format)
    return cpu_input, npu_input


class TestAllclose(TestCase):
    def cpu_op_exec(self, input_x, input_y):
        output = torch.allclose(input_x, input_y)
        return output

    def npu_op_exec(self, input_x, input_y):
        output = torch.allclose(input_x, input_y)
        return output

    def test_allclose_random(self, device):
        test_cases = [
            [[np.float32, -1, (1, 2)], [np.float32, -1, (1, 2)]],
            [[np.float32, -1, (1234, 2234)], [np.float32, -1, (1234, 2234)]],
            [[np.float32, -1, (321, 421, 521)], [np.float32, -1, (421, 521)]],
            [[np.float32, -1, (1, 600)], [np.float32, -1, (400, 200, 1)]],
            [[np.float32, -1, (20, 30, 40, 1)], [np.float32, -1, (30, 40, 50)]],
            [[np.float16, -1, (1, 2)], [np.float16, -1, (1, 2)]],
            [[np.float16, -1, (1234, 2234)], [np.float16, -1, (1234, 2234)]],
            [[np.float16, -1, (321, 421, 521)], [np.float16, -1, (421, 521)]],
            [[np.float16, -1, (1, 600)], [np.float16, -1, (400, 200, 1)]],
            [[np.float16, -1, (20, 30, 40, 1)], [np.float16, -1, (30, 40, 50)]],
            [[np.int8, -1, (1, 2)], [np.int8, -1, (1, 2)]],
            [[np.int8, -1, (1234, 2234)], [np.int8, -1, (1234, 2234)]],
            [[np.int8, -1, (321, 421, 521)], [np.int8, -1, (421, 521)]],
            [[np.int8, -1, (1, 600)], [np.int8, -1, (400, 200, 1)]],
            [[np.int8, -1, (20, 30, 40, 1)], [np.int8, -1, (30, 40, 50)]],
            [[np.uint8, -1, (1, 2)], [np.uint8, -1, (1, 2)]],
            [[np.uint8, -1, (1234, 2234)], [np.uint8, -1, (1234, 2234)]],
            [[np.int32, -1, (1, 2)], [np.int32, -1, (1, 2)]],
            [[np.int32, -1, (1234, 2234)], [np.int32, -1, (1234, 2234)]],
            [[np.int32, -1, (321, 421, 521)], [np.int32, -1, (421, 521)]],
            [[np.int32, -1, (1, 600)], [np.int32, -1, (400, 200, 1)]],
            [[np.int32, -1, (20, 30, 40, 1)], [np.int32, -1, (30, 40, 50)]],

        ]
        for item in test_cases:
            cpu_input_x, npu_input_x = create_common_tensor(item[0], 0, 100)
            cpu_input_y, npu_input_y = create_common_tensor(item[1], 0, 100)

            if cpu_input_x.dtype == torch.float16:
                cpu_input_x = cpu_input_x.to(torch.float32)

            if cpu_input_y.dtype == torch.float16:
                cpu_input_y = cpu_input_y.to(torch.float32)

            cpu_output = np.array(self.cpu_op_exec(cpu_input_x, cpu_input_y))
            npu_output = np.array(self.npu_op_exec(npu_input_x, npu_input_y))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_allclose_x_equal_y(self, device):
        test_cases = [
            [[np.float32, -1, (1, 2)], [np.float32, -1, (1, 2)]],
            [[np.float32, -1, (1234, 2234)], [np.float32, -1, (1234, 2234)]],
            [[np.float32, -1, (321, 421, 521)], [np.float32, -1, (421, 521)]],
            [[np.float32, -1, (1, 600)], [np.float32, -1, (400, 200, 1)]],
            [[np.float32, -1, (20, 30, 40, 1)], [np.float32, -1, (30, 40, 50)]],
            [[np.float16, -1, (1, 2)], [np.float16, -1, (1, 2)]],
            [[np.float16, -1, (1234, 2234)], [np.float16, -1, (1234, 2234)]],
            [[np.float16, -1, (321, 421, 521)], [np.float16, -1, (421, 521)]],
            [[np.float16, -1, (1, 600)], [np.float16, -1, (400, 200, 1)]],
            [[np.float16, -1, (20, 30, 40, 1)], [np.float16, -1, (30, 40, 50)]],
            [[np.int8, -1, (1, 2)], [np.int8, -1, (1, 2)]],
            [[np.int8, -1, (1234, 2234)], [np.int8, -1, (1234, 2234)]],
            [[np.int8, -1, (321, 421, 521)], [np.int8, -1, (421, 521)]],
            [[np.int8, -1, (1, 600)], [np.int8, -1, (400, 200, 1)]],
            [[np.int8, -1, (20, 30, 40, 1)], [np.int8, -1, (30, 40, 50)]],
            [[np.uint8, -1, (1, 2)], [np.uint8, -1, (1, 2)]],
            [[np.uint8, -1, (1234, 2234)], [np.uint8, -1, (1234, 2234)]],
            [[np.int32, -1, (1, 2)], [np.int32, -1, (1, 2)]],
            [[np.int32, -1, (1234, 2234)], [np.int32, -1, (1234, 2234)]],
            [[np.int32, -1, (321, 421, 521)], [np.int32, -1, (421, 521)]],
            [[np.int32, -1, (1, 600)], [np.int32, -1, (400, 200, 1)]],
            [[np.int32, -1, (20, 30, 40, 1)], [np.int32, -1, (30, 40, 50)]],
        ]
        for item in test_cases:
            cpu_input_x, npu_input_x = create_all_one_tensor(item[0])
            cpu_input_y, npu_input_y = create_all_one_tensor(item[1])

            if cpu_input_x.dtype == torch.float16:
                cpu_input_x = cpu_input_x.to(torch.float32)

            if cpu_input_y.dtype == torch.float16:
                cpu_input_y = cpu_input_y.to(torch.float32)

            cpu_output = np.array(self.cpu_op_exec(cpu_input_x, cpu_input_y))
            npu_output = np.array(self.npu_op_exec(npu_input_x, npu_input_y))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_allclose_scalar_1(self, device):
        input_x = np.array([1e-08]).astype(np.float32)
        input_y = np.array([1e-09]).astype(np.float32)
        cpu_input_x = torch.from_numpy(input_x)
        npu_input_x = torch.from_numpy(input_x).to("npu")
        cpu_input_y = torch.from_numpy(input_y)
        npu_input_y = torch.from_numpy(input_y).to("npu")

        cpu_output = np.array(self.cpu_op_exec(cpu_input_x, cpu_input_y))
        npu_output = np.array(self.npu_op_exec(npu_input_x, npu_input_y))
        self.assertRtolEqual(cpu_output, npu_output)

    def test_allclose_scalar_2(self, device):
        input_x = np.array([1e-07]).astype(np.float32)
        input_y = np.array([1e-08]).astype(np.float32)
        cpu_input_x = torch.from_numpy(input_x)
        npu_input_x = torch.from_numpy(input_x).to("npu")
        cpu_input_y = torch.from_numpy(input_y)
        npu_input_y = torch.from_numpy(input_y).to("npu")

        cpu_output = np.array(self.cpu_op_exec(cpu_input_x, cpu_input_y))
        npu_output = np.array(self.npu_op_exec(npu_input_x, npu_input_y))
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestAllclose, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:3")
    run_tests()
