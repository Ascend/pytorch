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


class TestIndexPut(TestCase):
    def cpu_op_exec(self, input1, indices, value):
        output = input1.index_put(indices, value)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, indices, value):
        output = input1.index_put(indices, value)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_exec(self, input1, indices, value):
        input1.index_put_(indices, value)
        output = input1.numpy()
        return output

    def npu_op_inp_exec(self, input1, indices, value):
        input1.index_put_(indices, value)
        input1 = input1.to("cpu")
        output = input1.numpy()
        return output

    def case_exec(self, shape):
        cpu_indices2 = []
        npu_indices2 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices2.append(cpu_indices1)
                npu_indices2.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_indices2, cpu_value)
            npu_output = self.npu_op_exec(npu_input, npu_indices2, npu_value)
            self.assertEqual(cpu_output, npu_output)

    def case_exec_fp16(self, shape):
        cpu_indices3 = []
        npu_indices3 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices3.append(cpu_indices1)
                npu_indices3.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_value = cpu_value.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_indices3, cpu_value)
            npu_output = self.npu_op_exec(npu_input, npu_indices3, npu_value)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def case_inp_exec(self, shape):
        cpu_indices4 = []
        npu_indices4 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices4.append(cpu_indices1)
                npu_indices4.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_inp_exec(
                cpu_input, cpu_indices4, cpu_value)
            npu_output = self.npu_op_inp_exec(
                npu_input, npu_indices4, npu_value)
            self.assertEqual(cpu_output, npu_output)

    def case_inp_exec_fp16(self, shape):
        cpu_indices5 = []
        npu_indices5 = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices5.append(cpu_indices1)
                npu_indices5.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_value = cpu_value.to(torch.float32)
            cpu_output = self.cpu_op_inp_exec(
                cpu_input, cpu_indices5, cpu_value)
            npu_output = self.npu_op_inp_exec(
                npu_input, npu_indices5, npu_value)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_index_put_shape_format_fp32(self, device="npu"):
        format_list = [0]
        shape_list = [(5, 6)]
        shape_format = [[[np.float32, i, j], [np.int64, 0, [1, 2]], [
            np.float32, 0, [1, 2]]] for i in format_list for j in shape_list]
        self.case_exec(shape_format)
        self.case_inp_exec(shape_format)

    def test_index_put_shape_format_fp16(self, device="npu"):
        format_list = [0]
        shape_list = [(5, 6)]
        shape_format = [[[np.float16, i, j], [np.int64, 0, [1, 2]], [
            np.float16, 0, [1, 2]]] for i in format_list for j in shape_list]
        self.case_exec_fp16(shape_format)
        self.case_inp_exec_fp16(shape_format)

    def test_index_put_null(self, device="npu"):
        cpu_input1 = torch.rand(2, 2)
        cpu_input2 = torch.rand(2, 2)
        cpu_mask_index = torch.tensor([[False, False], [False, False]])
        npu_mask_index = cpu_mask_index.to("npu")
        npu_input1 = cpu_input1.to("npu")
        npu_input2 = cpu_input2.to("npu")
        cpu_input1[cpu_mask_index] = cpu_input2.detach()[cpu_mask_index]
        npu_input1[npu_mask_index] = npu_input2.detach()[npu_mask_index]
        self.assertEqual(cpu_input1, npu_input1.to("cpu"))

    def test_index_put_undefined_fp32(self):
        cinput = torch.randn(4, 3, 2, 3)
        ninput = cinput.npu()
        cinput[:, [[1, 2, 1], [1, 2, 0]], :, [[1, 0, 2]]] = 1000
        ninput[:, [[1, 2, 1], [1, 2, 0]], :, [[1, 0, 2]]] = 1000
        self.assertRtolEqual(cinput.numpy(), ninput.cpu().numpy())

    def test_index_put_tensor_fp32(self):
        cinput = torch.randn(4, 4, 4, 4)
        ninput = cinput.npu()
        value = torch.tensor([100, 200, 300, 400], dtype = torch.float32)
        cinput[:, :, [0, 1, 2, 3], [0, 1, 2, 3]] = value
        ninput[:, :, [0, 1, 2, 3], [0, 1, 2, 3]] = value.npu()
        self.assertRtolEqual(cinput.numpy(), ninput.cpu().numpy())


if __name__ == "__main__":
    run_tests()
