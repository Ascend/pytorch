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


class TestIndexPut(TestCase):
    def cpu_op_exec(self, input, indices, value):
        output = input.index_put(indices, value)
        output = output.numpy()
        return output

    def npu_op_exec(self, input, indices, value):
        output = input.index_put(indices, value)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_exec(self, input, indices, value):
        input.index_put_(indices, value)
        output = input.numpy()
        return output

    def npu_op_inp_exec(self, input, indices, value):
        input.index_put_(indices, value)
        input = input.to("cpu")
        output = input.numpy()
        return output

    def case_exec(self, shape):
        cpu_indices = []
        npu_indices = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices.append(cpu_indices1)
                npu_indices.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_indices, cpu_value)
            npu_output = self.npu_op_exec(npu_input, npu_indices, npu_value)
            self.assertEqual(cpu_output, npu_output)

    def case_exec_fp16(self, shape):
        cpu_indices = []
        npu_indices = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices.append(cpu_indices1)
                npu_indices.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_value = cpu_value.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_indices, cpu_value)
            npu_output = self.npu_op_exec(npu_input, npu_indices, npu_value)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def case_inp_exec(self, shape):
        cpu_indices = []
        npu_indices = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices.append(cpu_indices1)
                npu_indices.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_inp_exec(
                cpu_input, cpu_indices, cpu_value)
            npu_output = self.npu_op_inp_exec(
                npu_input, npu_indices, npu_value)
            self.assertEqual(cpu_output, npu_output)

    def case_inp_exec_fp16(self, shape):
        cpu_indices = []
        npu_indices = []
        for item in shape:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input = cpu_input.to(torch.float32)
            for i in range(1, 3):
                cpu_indices1, npu_indices1 = create_common_tensor(
                    item[1], 1, 5)
                cpu_indices.append(cpu_indices1)
                npu_indices.append(npu_indices1)
            cpu_value, npu_value = create_common_tensor(item[2], 1, 100)
            cpu_value = cpu_value.to(torch.float32)
            cpu_output = self.cpu_op_inp_exec(
                cpu_input, cpu_indices, cpu_value)
            npu_output = self.npu_op_inp_exec(
                npu_input, npu_indices, npu_value)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_index_put_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [(5, 6)]
        shape_format = [[[np.float32, i, j], [np.int64, 0, [1, 2]], [
            np.float32, 0, [1, 2]]] for i in format_list for j in shape_list]
        self.case_exec(shape_format)
        self.case_inp_exec(shape_format)

    def test_index_put_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [(5, 6)]
        shape_format = [[[np.float16, i, j], [np.int64, 0, [1, 2]], [
            np.float16, 0, [1, 2]]] for i in format_list for j in shape_list]
        self.case_exec_fp16(shape_format)
        self.case_inp_exec_fp16(shape_format)


instantiate_device_type_tests(TestIndexPut, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()
