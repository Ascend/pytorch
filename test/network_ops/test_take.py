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

# coding: utf-8

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTake(TestCase):
    def cpu_op_out_exec(self, input1, input2, out):
        torch.take(input1, input2, out=out)
        output = out.numpy()
        return output

    def npu_op_out_exec(self, input1, input2, out):
        torch.take(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, input2):
        output = torch.take(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.take(input1, input2)
        output = output.to("cpu").numpy()
        return output

    def test_take_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (5, 3)], [np.int64, 0, (3)], 8],
            [[np.int8, 0, (64, 10)], [np.int64, 0, (10)], 74],
            [[np.uint8, -1, (256, 2048, 7, 7)], [np.int64, -1, (30)], 2748],
            [[np.int16, -1, (32, 1, 3, 3)], [np.int64, -1, (32)], 39],
            [[np.int64, -1, (10, 128)], [np.int64, -1, (128)], 138],
            [[np.float16, 0, (64, 10)], [np.int64, 0, (10)], 74],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, item[2])
            if item[0][0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            if npu_input1.dtype == torch.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_take_out_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (5, 3)], [np.int64, 0, (3)], 8, [np.float32, 0, (3)]],
            [[np.int8, 0, (64, 10)], [np.int64, 0, (10)], 74, [np.int8, 0, (10)]],
            [[np.uint8, -1, (256, 2048, 7, 7)], [np.int64, -1, (30)], 2748, [np.uint8, -1, (30)]],
            [[np.int16, -1, (32, 1, 3, 3)], [np.int64, -1, (32)], 39, [np.int16, -1, (32)]],
            [[np.int64, -1, (10, 128)], [np.int64, -1, (128)], 138, [np.int64, -1, (128)]],
            [[np.float16, 0, (64, 10)], [np.int64, 0, (10)], 74, [np.float16, 0, (10)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output1, npu_output1 = create_common_tensor(item[3], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, item[2])
            if item[0][0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_output1 = cpu_output1.to(torch.float32)
            cpu_output = self.cpu_op_out_exec(cpu_input1, cpu_input2, cpu_output1)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, npu_output1)
            if npu_input1.dtype == torch.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
