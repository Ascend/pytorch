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


class TestExp(TestCase):
    def cpu_op_exec(self, input):
        output = torch.exp(input)
        output = output.numpy()
        return output

    def npu_op_exec(self, input):
        output = torch.exp(input)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_out_exec(self, input, output):
        torch.exp(input, out = output)
        output = output.to("cpu").numpy()
        return output

    def test_exp_shape_format_fp16(self, device):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            # cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp_out_float32_shape_format(self, device):
        shape_format = [
            [[np.float32, 0, [1024, 32, 7, 7]], [np.float32, 0, [1024, 32, 7, 7]]],
            [[np.float32, 0, [1024, 32, 7]], [np.float32, 0, [1024, 32]]],
            [[np.float32, 0, [1024, 32]], [np.float32, 0, [1024, 32]]],
            [[np.float32, 0, [1024]], [np.float32, 0, [1024, 1]]],
            [[np.float32, 3, [1024, 32, 7, 7]], [np.float32, 3, [1024, 32, 7, 7]]],
            [[np.float32, 3, [1024, 32, 7]], [np.float32, 3, [1024, 32]]],
            [[np.float32, 3, [1024, 32]], [np.float32, 3, [1024, 20]]],
            [[np.float32, 3, [1024]], [np.float32, 3, [1024]]],
            ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_output, npu_output = create_common_tensor(item[1], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input, npu_output)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp_out_float16_shape_format(self, device):
        shape_format = [
            [[np.float16, 0, [1024, 32, 7, 7]], [np.float16, 0, [1024, 32, 7, 7]]],
            [[np.float16, 0, [1024, 32, 7]], [np.float16, 0, [1024, 32]]],
            [[np.float16, 0, [1024, 32]], [np.float16, 0, [1024, 32]]],
            [[np.float16, 0, [1024]], [np.float16, 0, [1024, 1]]],
            [[np.float16, 3, [1024, 32, 7, 7]], [np.float16, 3, [1024, 32, 7, 7]]],
            [[np.float16, 3, [1024, 32, 7]], [np.float16, 3, [1024, 32]]],
            [[np.float16, 3, [1024, 32]], [np.float16, 3, [1024, 20]]],
            [[np.float16, 3, [1024]], [np.float16, 3, [1024]]],
            ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_output, npu_output = create_common_tensor(item[1], -1, 1)
            if item[0][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input, npu_output)
            if item[0][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestExp, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
