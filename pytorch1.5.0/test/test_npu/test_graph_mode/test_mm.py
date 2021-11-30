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
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests
from graph_utils import graph_mode

class TestMatMul(TestCase):
    def trans_tensor(self, mat1, mat2):
        if mat1.size(1) == mat2.size(0):
            return mat1, mat2
        mat = mat1.t()
        if mat.size(1) == mat2.size(0):
            return mat, mat2
        mat = mat2.t()
        if mat1.size(1) == mat.size(0):
            return mat1, mat
        return mat1, mat2

    def cpu_op_exec(self, input1, input2):
        output = torch.mm(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.mm(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def mm_auto_list_exec(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input_1, cpu_input_2 = self.trans_tensor(cpu_input1, cpu_input2)
            npu_input_1, npu_input_2 = self.trans_tensor(npu_input1, npu_input2)
            cpu_output = self.cpu_op_exec(cpu_input_1, cpu_input_2)
            npu_output = self.npu_op_exec(npu_input_1, npu_input_2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_muls_shape_format_fp16(self, device):
        format_list = [0, 3, 29]
        shape_list = [[1024, 1000], [1000, 1024],
                      [1024, 1024]]
        shape_format1 = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        shape_format2 = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        shape_format = [
            [i, j] for i in shape_format1 for j in shape_format2
        ]
        self.mm_auto_list_exec(shape_format)

    @graph_mode
    def test_matmul_shape_format_fp32(self, device):
        format_list = [0, 3, 29]
        shape_list = [[256, 1280], [1000, 1280],
        ]
        shape_format1 = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        shape_format2 = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        shape_format = [
            [i, j] for i in shape_format1 for j in shape_format2
        ]
        self.mm_auto_list_exec(shape_format)

instantiate_device_type_tests(TestMatMul, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()
