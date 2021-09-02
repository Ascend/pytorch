# Copyright (c) 2021 Huawei Technologies Co., Ltd
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

from torch._C import dtype
import torch
import torch.nn.functional as F
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor

class TestTriangularSolveHelper(TestCase):
    def cpu_op_exec(self, input1, input2, upper, transpose, unitriangular):
        output_s, output_a = input1.triangular_solve(input2, upper, transpose, unitriangular)
        return output_s, output_a

    def npu_op_exec(self, input1, input2, upper, transpose, unitriangular):
        output_s, output_a = input1.triangular_solve(input2, upper, transpose, unitriangular)
        output_s = output_s.cpu()
        output_a = output_a.cpu()
        return output_s, output_a

    def test_triangular_solve_helper_fp32(self, device):
        shape_format = [
            [[np.float32, -1, [2, 3]], [np.float32, -1, [2, 2]]],
            [[np.float32, -1, [3, 2, 3]], [np.float32, -1, [3, 2, 2]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            for upper in [True, False]:
                for transpose in [True, False]:
                    for unitriangular in [True, False]:
                        cpu_s, cpu_a = self.cpu_op_exec(cpu_input1, cpu_input2, upper, transpose, unitriangular)
                        npu_s, npu_a = self.npu_op_exec(npu_input1, npu_input2, upper, transpose, unitriangular)
                        self.assertRtolEqual(cpu_a, npu_a)
                        self.assertRtolEqual(cpu_s, npu_s)

    def test_triangular_solve_helper_out(self, device):
        input1 = torch.randn(2, 3, 4, 5, dtype = torch.float32)
        A = torch.randn(4, 4, dtype = torch.float32)
        r = torch.randn(2, 3, 4, 5, dtype = torch.float32)
        c_a = torch.randn(4, 4, dtype = torch.float32)
        npu_input = input1.npu()
        An = A.npu()
        rn = torch.randn(2, 3, 4, 5, dtype = torch.float32)
        c_an = torch.randn(4, 4, dtype = torch.float32)
        cout = torch.triangular_solve(input1, A, out = (r, c_a))
        nout = torch.triangular_solve(npu_input, An, out = (rn, c_an))
        self.assertRtolEqual(r, rn)
        self.assertRtolEqual(c_a, c_an)

instantiate_device_type_tests(TestTriangularSolveHelper, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()

