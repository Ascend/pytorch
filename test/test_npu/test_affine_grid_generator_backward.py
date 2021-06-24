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
import sys
from torch.nn import functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestAffineGridGeneratorBackward(TestCase):
    def test_affine_grid_generator_backward_common_shape(self, device):
        shape_list = [[100, 2, 3], [10, 2, 3]]
        shape_format = [
            [np.float32, -1, j] for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            size = torch.Size((item[2][0], 2, 28, 2))
            cpu_input.requires_grad = True
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_input.requires_grad = True
            npu_output = self.npu_op_exec(npu_input, size)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_affine_grid_generator_backward_fp16(self, device):
        shape_list = [[100, 2, 3], [10, 2, 3]]
        shape_format = [
            [np.float16, -1, j] for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_input = cpu_input.to(torch.float32)
            npu_input = npu_input.to(torch.float32)
            size = torch.Size((item[2][0], 2, 28, 2))
            cpu_input.requires_grad = True
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_input.requires_grad = True
            npu_output = self.npu_op_exec(npu_input, size)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output.astype(np.float16))
    
    def cpu_op_exec(self, input, size):
        out = F.affine_grid(input, size, True)
        input.requires_grad = True
        grad_output = torch.ones(out.size(), dtype=torch.float)
        out.backward(gradient=grad_output)
        output = input.grad.numpy()
        return output

    def npu_op_exec(self, input, size):
        input.requires_grad = True
        out = F.affine_grid(input, size, True)
        grad_output = torch.ones(out.size(), dtype=torch.float).npu()
        out.backward(gradient=grad_output)
        output = input.grad.to("cpu").numpy()
        return output

instantiate_device_type_tests(TestAffineGridGeneratorBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()