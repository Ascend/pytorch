# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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
from graph_utils import graph_mode


class TestAddmm(TestCase):
    def generate_scalar(self, dtype, min_num, max_num):
        if dtype == "float32":
            scalar = np.random.uniform(min_num, max_num)
        if dtype == "int32":
            scalar = np.random.randint(min_num, max_num)
        return scalar

    def cpu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.addmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addmm(input1, input2, input3, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar1, scalar2, input4):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = input4.to("npu")
        torch.addmm(input1, input2, input3, beta=scalar1, alpha=scalar2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input1.addmm_(input2, input3, beta=scalar1, alpha=scalar2)
        output = input1.to("cpu")
        output = output.numpy()
        return output


    def cpu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input3_t = input3.t()
        output = torch.addmm(input1, input2, input3_t, beta=scalar1, alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input3_t = input3.t()
        output = torch.addmm(input1, input2, input3_t, beta=scalar1, alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @graph_mode
    def test_addmm(self, device):
        shape_format = [
            [[np.float32, 0, [3, 3]], [np.float32, 0, [3, 5]], [np.float32, 0, [5, 3]], "float32"],
            [[np.int32, 0, [3, 3]], [np.int32, 0, [3, 5]], [np.int32, 0, [5, 3]], "int32"]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 100)
            cpu_input4, npu_input4 = create_common_tensor(item[0], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            npu_output1 = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, scalar1, scalar2, npu_input4)
            npu_output2 = self.npu_op_exec_inplace(npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output1)
            self.assertRtolEqual(cpu_output, npu_output2)

    @graph_mode
    def test_addmm_transpose(self, device):
        shape_format = [
            [[np.float32, 0, [4, 5]], [np.float32, 0, [4, 7]], [np.float32, 0, [5, 7]], "float32"],
            [[np.int32, 0, [4, 5]], [np.int32, 0, [4, 7]], [np.int32, 0, [5, 7]], "int32"]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_transpose_output = self.npu_op_transpose_exec(npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, npu_transpose_output)


instantiate_device_type_tests(TestAddmm, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
