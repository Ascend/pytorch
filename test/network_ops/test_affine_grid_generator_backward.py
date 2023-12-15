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
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAffineGridGeneratorBackward(TestCase):
    def test_affine_grid_generator_backward_common_shape(self, device="npu"):
        shape_list = [[100, 2, 3], [10, 2, 3]]
        shape_format = [
            [np.float32, -1, j] for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 1)
            size = torch.Size((item[2][0], 2, 28, 2))
            cpu_input1.requires_grad = True
            cpu_output = self.cpu_op_exec(cpu_input1, size)
            npu_input1.requires_grad = True
            npu_output = self.npu_op_exec(npu_input1, size)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_affine_grid_generator_backward_fp16(self, device="npu"):
        shape_list = [[100, 2, 3], [10, 2, 3]]
        shape_format = [
            [np.float16, -1, j] for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 1)
            cpu_input1 = cpu_input1.to(torch.float32)
            npu_input1 = npu_input1.to(torch.float32)
            size = torch.Size((item[2][0], 2, 28, 2))
            cpu_input1.requires_grad = True
            cpu_output = self.cpu_op_exec(cpu_input1, size)
            npu_input1.requires_grad = True
            npu_output = self.npu_op_exec(npu_input1, size)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output.astype(np.float16))

    def cpu_op_exec(self, input1, size):
        out = F.affine_grid(input1, size, True)
        input1.requires_grad = True
        grad_output = torch.ones(out.size(), dtype=torch.float)
        out.backward(gradient=grad_output)
        output = input1.grad.numpy()
        return output

    def npu_op_exec(self, input1, size):
        input1.requires_grad = True
        out = F.affine_grid(input1, size, True)
        grad_output = torch.ones(out.size(), dtype=torch.float).npu()
        out.backward(gradient=grad_output)
        output = input1.grad.to("cpu").numpy()
        return output


if __name__ == "__main__":
    run_tests()
