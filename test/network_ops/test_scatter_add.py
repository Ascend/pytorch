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


class TestScatterAdd(TestCase):
    def cpu_op_exec_inp(self, input1, dim, index, src):
        input1.scatter_add_(dim, index, src)
        output = input1.numpy()
        return output

    def npu_op_exec_inp(self, input1, dim, index, src):
        input1.scatter_add_(dim, index, src)
        input1 = input1.to("cpu")
        output = input1.numpy()
        return output

    def cpu_op_exec(self, input1, dim, index, src):
        output = torch.scatter_add(input1, dim, index, src)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim, index, src):
        output = torch.scatter_add(input1, dim, index, src)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_scatter_add_common_shape_format(self, device="npu"):
        shape_format = [
            [0, [np.int64, 0, [10, 20]], [np.float32, 0, [10, 20]], [np.float32, 0, [10, 20]]],
            [1, [np.int64, 0, [10, 20]], [np.float32, 0, [10, 20]], [np.float32, 0, [10, 20]]],
            [0, [np.int64, 0, [2, 6]], [np.float32, 0, [2, 6]], [np.float32, 0, [2, 6]]],
            [1, [np.int64, 0, [2, 6]], [np.float32, 0, [2, 6]], [np.float32, 0, [2, 6]]],
            [0, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
            [1, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
            [2, [np.int64, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]], [np.float32, 0, [10, 20, 30]]],
        ]

        for item in shape_format:
            cpu_input2, npu_input2 = create_common_tensor(item[2], 1, 100)
            cpu_input1, npu_input1 = create_common_tensor(item[1], 1, (item[1][2][item[0]] - 1))
            cpu_input3, npu_input3 = create_common_tensor(item[3], 1, 100)

            cpu_output = self.cpu_op_exec(cpu_input3, item[0], cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input3, item[0], npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_inp_output = self.cpu_op_exec_inp(cpu_input3, item[0], cpu_input1, cpu_input2)
            npu_inp_output = self.npu_op_exec_inp(npu_input3, item[0], npu_input1, npu_input2)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)

    def test_scatter_add_float16_shape_format(self, device="npu"):
        def cpu_op_exec_inp_fp16(input1, dim, index, src):
            input1 = input1.to(torch.float32)
            src = src.to(torch.float32)
            input1.scatter_add_(dim, index, src)
            output = input1.numpy()
            output = output.astype(np.float16)
            return output

        def cpu_op_exec_fp16(input1, dim, index, src):
            output = torch.scatter_add(input1, dim, index, src)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [0, [np.int64, 0, [10, 20]], [np.float16, 0, [10, 20]], [np.float16, 0, [10, 20]]],
            [1, [np.int64, 0, [10, 20]], [np.float16, 0, [10, 20]], [np.float16, 0, [10, 20]]],
            [0, [np.int64, 0, [2, 6]], [np.float16, 0, [2, 6]], [np.float16, 0, [2, 6]]],
            [1, [np.int64, 0, [2, 6]], [np.float16, 0, [2, 6]], [np.float16, 0, [2, 6]]],
            [0, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
            [1, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
            [2, [np.int64, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]], [np.float16, 0, [10, 20, 30]]],
        ]

        for item in shape_format:
            cpu_input2, npu_input2 = create_common_tensor(item[2], 1, 100)
            cpu_input1, npu_input1 = create_common_tensor(item[1], 1, (item[1][2][item[0]] - 1))
            cpu_input3, npu_input3 = create_common_tensor(item[3], 1, 100)

            cpu_output = cpu_op_exec_fp16(cpu_input3, item[0], cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input3, item[0], npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
            cpu_inp_output = cpu_op_exec_inp_fp16(cpu_input3, item[0], cpu_input1, cpu_input2)
            npu_inp_output = self.npu_op_exec_inp(npu_input3, item[0], npu_input1, npu_input2)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)


if __name__ == "__main__":
    run_tests()
