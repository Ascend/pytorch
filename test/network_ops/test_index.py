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


class TestIndex(TestCase):

    def generate_index_data_bool(self, shape):
        cpu_input = torch.randn(shape) > 0
        npu_input = cpu_input.to("npu")
        return cpu_input, npu_input

    def cpu_op_exec(self, input1, index):
        output = input1[index]
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, index):
        output = input1[index]
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_ellip(self, input1, index):
        output = input1[index, ..., index]
        output = output.numpy()
        return output

    def npu_op_exec_ellip(self, input1, index):
        output = input1[index, ..., index]
        output = output.cpu().numpy()
        return output

    def cpu_op_exec_semi(self, input1, index):
        output = input1[index, :, index]
        output = output.numpy()
        return output

    def npu_op_exec_semi(self, input1, index):
        output = input1[index, :, index]
        output = output.cpu().numpy()
        return output

    def test_index_ellip(self):
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[5, 256, 256, 100]]
        shape_format_tensor = [
            [[i, j, k], [np.int64, 0, (1, 2)]] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_tensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 0, 2)
            cpu_output = self.cpu_op_exec_ellip(cpu_input1, cpu_index1)
            npu_output = self.npu_op_exec_ellip(npu_input1, npu_index1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_semi(self):
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[5, 256, 256, 100]]
        shape_format_tensor = [
            [[i, j, k], [np.int64, 0, (1, 2)]] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_tensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 0, 2)
            cpu_output = self.cpu_op_exec_semi(cpu_input1, cpu_index1)
            npu_output = self.npu_op_exec_semi(npu_input1, npu_index1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_tensor(self):
        # test index is tensor
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_tensor = [
            [[i, j, k], [np.int64, 0, (1, 2)]] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_tensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 1, 3)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_index1)
            npu_output = self.npu_op_exec(npu_input1, npu_index1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_tensor_x(self):
        # 注：test index is [tensor, x] , (x=1,bool,range)
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_tensor = [
            [[i, j, k], [np.int64, 0, (1, 2)]] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_tensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 1, 3)
            for i in [1, range(2), True]:
                cpu_output = self.cpu_op_exec(cpu_input1, (cpu_index1, i))
                npu_output = self.npu_op_exec(npu_input1, (npu_index1, i))
                self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_tensor_tensor(self):
        # test index is [tensor, tensor]
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 1000]]
        shape_format_multiTensor = [
            [[i, j, k], [np.int64, 0, [1, 2]]] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_multiTensor:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index1, npu_index1 = create_common_tensor(item[1], 1, 3)
            cpu_index2, npu_index2 = create_common_tensor(item[1], 1, 3)
            cpu_output = self.cpu_op_exec(cpu_input1, (cpu_index1, cpu_index2))
            npu_output = self.npu_op_exec(npu_input1, (npu_index1, npu_index2))
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_list(self):
        # test index is list
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_list = [
            [[i, j, k], (0, 1)] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_list:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_list_x(self):
        # 注：test index is [list, x],  (x=1,bool,range)
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_list = [
            [[i, j, k], (0, 1)] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_list:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            for i in [1, range(2), (0, 1), True]:
                cpu_output = self.cpu_op_exec(cpu_input1, (item[1], i))
                npu_output = self.npu_op_exec(npu_input1, (item[1], i))
                self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_tensor_bool(self):
        # 注：test index is bool tensor
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        shape_format_tensor_bool = [
            [[i, j, k], k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in shape_format_tensor_bool:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_index, npu_index = self.generate_index_data_bool(item[1])
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_index)
            npu_output = self.npu_op_exec(npu_input1, npu_index)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_shape_format_bool_x(self):
        # 注：test index is [bool, x] , (x=1,bool,range)
        dtype_list = [np.float32, np.float16, np.int32]
        format_list = [0]
        shape_list = [[256, 10], [256, 256, 100], [5, 256, 256, 100]]
        index_list = [(True), (False), (True, 1),
                      (True, range(4)), (True, False)]
        shape_format_tensor_bool_list = [
            [[i, j, k], l] for i in dtype_list for j in format_list for k in shape_list for l in index_list
        ]

        for item in shape_format_tensor_bool_list:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_aicore_shape_format(self):
        format_shape = [
            [[np.float32, 0, 1], [np.int64, 0, 29126]],
            [[np.float32, 0, (1, 4)], [np.int64, 0, 29126]],
            [[np.float32, 0, (1, 29126, 10)], [np.bool, 0, (1, 29126, 10)]],
            [[np.float32, 0, (8400, 16)], [np.bool, 0, (8400, 16)]],
            [[np.float32, 0, (8400, 16)], [np.bool, 0, 8400]],
            [[np.bool, 0, (8400, 16)], [np.int64, 0, 1237]],
            [[np.bool, 0, (8400, 18)], [np.int64, 0, 2999]],
        ]

        for item in format_shape:
            cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
            if item[1][0] == np.bool:
                cpu_index, npu_index = self.generate_index_data_bool(item[1][2])
            else:
                cpu_index, npu_index = create_common_tensor(item[1], 0, cpu_input.dim() - 1)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_index)
            npu_output = self.npu_op_exec(npu_input, npu_index)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_index_different_device(self):
        index = torch.tensor([[True, True], [False, False]])
        cpu_input1 = torch.rand([2, 2], dtype=torch.float32)
        npu_input1 = cpu_input1.npu()
        cpu_output1 = self.cpu_op_exec(cpu_input1, index)
        npu_output1 = self.npu_op_exec(npu_input1, index)
        self.assertRtolEqual(cpu_output1, npu_output1)

        cpu_input2 = torch.randn(15, 5)
        cpu_index2 = torch.randn(15, 5) > 0.5
        npu_index2 = cpu_index2.npu()
        cpu_output2 = self.cpu_op_exec(cpu_input2, cpu_index2)
        npu_output2 = self.npu_op_exec(cpu_input2, npu_index2)
        self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
