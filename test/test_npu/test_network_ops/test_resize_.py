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
import itertools
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests


class TestResize(TestCase):
    def cpu_op_exec(self, cpu_in, cpu_out, shape, op):
        cpu_out.resize_(shape)
        op(cpu_in, cpu_in, out=cpu_out)
        return cpu_out

    def npu_op_exec(self, npu_in, npu_out, shape, op):
        npu_out.resize_(shape)
        op(npu_in, npu_in, out=npu_out)
        return npu_out

    def op_result_cmp_(self, shape_a, shape_b, op, is_contiguous=False):
        a = torch.rand(shape_a)
        b = torch.full(shape_b, 100.)
        if is_contiguous:
            b = b.t()
        cpu = self.cpu_op_exec(a, b, shape_a, op)

        nb = torch.full(shape_b, 100.)
        a_npu = a.npu()
        b_npu = nb.npu()
        if is_contiguous:
            b_npu = b_npu.t()
        npu = self.npu_op_exec(a_npu, b_npu, shape_a, op)

        cpu.add_(10)
        npu.add_(10)
        self.assertRtolEqual(cpu.numpy(), npu.cpu().numpy())

    def test_op_resize_(self, device):
        operators = [torch.add, torch.mul, torch.matmul]
        shape_a = (5, 5)
        contiguous = [True, False]

        smalls = [(0, ), (1, ), (3, 1), (2, 3)]
        for shape_b, op, is_contiguous in itertools.product(smalls, operators, contiguous):
            self.op_result_cmp_(shape_a, shape_b, op, is_contiguous)

        bigs = [(10, 9), (11, 11), (8, 11)]
        for shape_b, op, is_contiguous in itertools.product(bigs, operators, contiguous):
            self.op_result_cmp_(shape_a, shape_b, op, is_contiguous)


instantiate_device_type_tests(TestResize, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()