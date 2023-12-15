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

import random
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIndexPut(TestCase):
    def cpu_op_exec(self, input1, indices, value):
        output = input1.index_put(indices, value, accumulate=True)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, indices, value):
        output = input1.index_put(indices, value, accumulate=True)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_exec(self, input1, indices, value):
        input1.index_put_(indices, value, accumulate=True)
        output = input1.numpy()
        return output

    def npu_op_inp_exec(self, input1, indices, value):
        input1.index_put_(indices, value, accumulate=True)
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
        value = torch.tensor([100, 200, 300, 400], dtype=torch.float32)
        cinput[:, :, [0, 1, 2, 3], [0, 1, 2, 3]] = value
        ninput[:, :, [0, 1, 2, 3], [0, 1, 2, 3]] = value.npu()
        self.assertRtolEqual(cinput.numpy(), ninput.cpu().numpy())

    def test_index_put_NCDHW_and_resize(self):
        index_list = [20, 26, 21, 8, 25], [8, 15, 2, 17, 1], [20, 79, 5, 28, 45], [24, 3, 44, 37, 10], [2, 3, 0, 10, 14]
        index = [torch.LongTensor(i) for i in index_list]
        input_data = torch.randn(32, 20, 80, 80, 16)
        input2 = torch.randn(5,)

        index_npu = [i.npu() for i in index]
        input_data_npu = input_data.npu()
        input2_npu = input2.npu()

        input_data.index_put_(index, input2)
        cpu_res = input_data.numpy()
        input_data_npu.index_put_(index_npu, input2_npu)
        npu_res = input_data_npu.cpu().numpy()
        self.assertRtolEqual(cpu_res, npu_res)

    def test_index_put_different_device(self):
        m = random.randint(10, 20)
        elems = random.randint(20000, 30000)
        cpu_values = torch.rand(elems)
        npu_values = cpu_values.npu()
        indices = torch.randint(m, (elems,))
        cpu_input1 = torch.rand(m)
        npu_input1 = cpu_input1.npu()
        cpu_output1 = self.cpu_op_exec(cpu_input1, (indices,), cpu_values)
        npu_output1 = self.npu_op_exec(npu_input1, (indices,), npu_values)
        self.assertRtolEqual(cpu_output1, npu_output1)

        m = random.randint(10, 20)
        elems = random.randint(20000, 30000)
        values = torch.rand(elems)
        cpu_indices = torch.randint(m, (elems,))
        npu_indices = cpu_indices.npu()
        input1 = torch.rand(m)
        cpu_output2 = self.cpu_op_exec(input1, (cpu_indices,), values)
        npu_output2 = self.npu_op_exec(input1, (npu_indices,), values)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_index_put_dim_size(self):
        npu_input1 = torch.arange(0, 4).npu()
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            npu_input1[npu_input1 > -1] = torch.tensor([1, 2, 3]).npu()


if __name__ == "__main__":
    run_tests()
