# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCountNonzero(TestCase):
    def npu_op_exec(self, input1):
        output = torch.count_nonzero(input1).cpu().numpy()
        return output
    
    def cpu_op_exec(self, input1):
        output = torch.count_nonzero(input1).numpy()
        return output
    
    def npu_op_exec_dim(self, input1, dim):
        output = torch.count_nonzero(input1, dim).cpu().numpy()
        return output
    
    def cpu_op_exec_dim(self, input1, dim):
        output = torch.count_nonzero(input1, dim).numpy()
        return output

    def count_nonzero_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def count_nonzero_dim_result(self, shape_format):
        for item in shape_format:
            dim = np.random.randint(0, len(item[2]))
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output_dim = self.cpu_op_exec_dim(cpu_input, dim=[0])
            npu_output_dim = self.npu_op_exec_dim(npu_input, dim=[0])
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

            cpu_output_dim = self.cpu_op_exec_dim(cpu_input, dim)
            npu_output_dim = self.npu_op_exec_dim(npu_input, dim)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)

    def test_count_nonzero_fp32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_result(shape_format)

    def test_count_nonzero_dim_fp32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_dim_result(shape_format)

    def test_count_nonzero_fp16(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_result(shape_format)

    def test_count_nonzero_dim_fp16(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_dim_result(shape_format)

    def test_count_nonzero_int32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.int32, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_result(shape_format)

    def test_count_nonzero_dim_int32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.int32, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_dim_result(shape_format)

    def test_count_nonzero_bool(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.bool_, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_result(shape_format)

    def test_count_nonzero_dim_bool(self):
        format_list = [0, 3, 4, 29]
        shape_list = [[18], [256, 100], [32, 48, 64], [32, 24, 18, 18]]
        shape_format = [
            [np.bool_, i, j] for i in format_list for j in shape_list
        ]
        self.count_nonzero_dim_result(shape_format)


if __name__ == "__main__":
    run_tests()

