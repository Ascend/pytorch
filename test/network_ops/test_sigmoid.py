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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestSigmoid(TestCase):
    @Dtypes(torch.float)
    def test_sigmoid(self, dtype, device):
        inputValues = [-1000, -1, 0, 0.5, 1, 2, 1000]
        expectedOutput = [0.0000, 0.2689, 0.5, 0.6225, 0.7311, 0.8808, 1.000]
        precision_4dps = 0.0002

        self.assertEqual(
            torch.tensor(
                inputValues, dtype=dtype, device=device).sigmoid().cpu(),
            torch.tensor(
                expectedOutput,
                dtype=dtype, device=device).cpu(), precision_4dps)

    def cpu_op_exec(self, input1):
        output = torch.sigmoid(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.sigmoid(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_out_exec(self, input1, output):
        torch.sigmoid(input1, out=output)
        output = output.to("cpu").numpy()
        return output

    def test_sigmoid_shape_format_fp16(self, device="npu"):
        format_list = [0]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sigmoid_shape_format_fp32(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_list = [1, (32, 32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sigmoid_out_float32_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, [1024, 32, 7, 7]], [np.float32, 0, [1024, 32, 7, 7]]],
            [[np.float32, 0, [1024, 32, 7]], [np.float32, 0, [128, 32, 8, 10]]],
            [[np.float32, 0, [512, 32]], [np.float32, 0, [1024, 20]]],
            [[np.float32, 0, [1024]], [np.float32, 0, [1024, 1]]],
            [[np.float32, 3, [1024, 32, 7, 7]], [np.float32, 3, [1024, 32, 7, 7]]],
            [[np.float32, 3, [1024, 32, 7]], [np.float32, 3, [1024, 32]]],
            [[np.float32, 3, [1024, 32]], [np.float32, 3, [1024, 20]]],
            [[np.float32, 3, [1024]], [np.float32, 3, [1024]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output, npu_output = create_common_tensor(item[1], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input, npu_output)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sigmoid_out_float16_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, 0, [1024, 32, 7, 7]], [np.float16, 0, [1024, 32, 7, 7]]],
            [[np.float16, 0, [1024, 32, 7]], [np.float16, 0, [128, 32, 8, 10]]],
            [[np.float16, 0, [510, 32]], [np.float16, 0, [1024, 20]]],
            [[np.float16, 0, [1024]], [np.float16, 0, [1024, 1]]],
            [[np.float16, 3, [1024, 32, 7, 7]], [np.float16, 3, [1024, 32, 7, 7]]],
            [[np.float16, 3, [1024, 32, 7]], [np.float16, 3, [1024, 32]]],
            [[np.float16, 3, [1024, 32]], [np.float16, 3, [1024, 20]]],
            [[np.float16, 3, [1024]], [np.float16, 3, [128]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output, npu_output = create_common_tensor(item[1], -1, 1)
            if item[0][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input, npu_output)
            if item[0][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
