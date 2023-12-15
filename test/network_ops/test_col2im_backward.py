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


class TestCol2ImBackward(TestCase):

    def cpu_op_exec(self, input1, output_size, ksizes, strides, dilates, padding):
        input1.requires_grad = True
        output = torch._C._nn.col2im(input1, output_size, ksizes, dilates, padding, strides)
        output.backward(torch.ones_like(output))
        output1 = output.detach().numpy()
        cpu_grad = input1.grad
        return output1, cpu_grad.detach().numpy()

    def npu_op_exec(self, input1, output_size, ksizes, strides, dilates, padding):
        input1.requires_grad = True
        output = torch._C._nn.col2im(input1, output_size, ksizes, dilates, padding, strides)
        output.backward(torch.ones_like(output))
        output1 = output.detach().cpu().numpy()
        npu_grad = input1.grad
        return output1, npu_grad.detach().cpu().numpy()

    def test_col2imbackward_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (4, 12, 12)], (4, 5), (2, 2), (1, 1), (1, 1), (0, 0)],
            [[np.float16, 0, (12, 18, 9)], (4, 5), (2, 3), (1, 1), (1, 1), (0, 0)],
            [[np.float16, 0, (1, 24, 42)], (7, 8), (2, 2), (1, 1), (1, 1), (0, 0)]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 20)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4], item[5])
            npu_output, npu_grad = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
