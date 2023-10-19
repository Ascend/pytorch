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
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestKlDivBackward(TestCase):
    def cpu_op_exec(self, input1, input2, reduction):
        input1.requires_grad = True
        output = torch.kl_div(input1, input2, reduction=reduction)
        output.backward(torch.ones_like(output))
        output = output.detach().numpy()
        return output, input1.grad

    def npu_op_exec(self, input1, input2, reduction):
        input1.requires_grad = True
        output = torch.kl_div(input1, input2, reduction=reduction)
        output.backward(torch.ones_like(output))
        output = output.cpu()
        output = output.detach().numpy()
        return output, input1.grad

    def test_kl_div_backward_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[torch.float16, 0, (192, 8)], [torch.float16, 0, (192, 8)], 1],
            [[torch.float16, 0, (192, 50000)], [torch.float16, 0, (192, 50000)], 1],
            [[torch.float16, 0, (2, 2)], [torch.float16, 0, (2, 2)], 2],
            [[torch.float16, 0, (3, 5)], [torch.float16, 0, (3, 5)], 0],
            [[torch.float16, 0, (2, 4, 3)], [torch.float16, 0, (2, 4, 3)], 0],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input = F.log_softmax(x, dim=0)
            cpu_target = F.softmax(y, dim=0)
            npu_input = cpu_input.npu()
            npu_target = cpu_target.npu()
            reduction = item[2]
            cpu_output, cpu_input_grad = self.cpu_op_exec(cpu_input, cpu_target, reduction)
            npu_output, npu_input_grad = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad.cpu())

    def test_kl_div_backward_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[torch.float16, 0, (112, 8)], [torch.float16, 0, (112, 8)], 1],
            [[torch.float16, 0, (112, 50000)], [torch.float16, 0, (112, 50000)], 1],
            [[torch.float16, 0, (2, 3)], [torch.float16, 0, (2, 3)], 2],
            [[torch.float16, 0, (3, 6)], [torch.float16, 0, (3, 6)], 0],
            [[torch.float16, 0, (2, 3, 3)], [torch.float16, 0, (2, 3, 3)], 0],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input = F.log_softmax(x, dim=0).to(item[0][0])
            cpu_target = F.softmax(y, dim=0).to(item[1][0])
            npu_input = cpu_input.npu()
            npu_target = cpu_target.npu()
            reduction = item[2]
            cpu_output, cpu_input_grad = self.cpu_op_exec(
                cpu_input.to(torch.float32),
                cpu_target.to(torch.float32),
                reduction)
            npu_output, npu_input_grad = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)
            self.assertRtolEqual(cpu_input_grad.to(torch.float16), npu_input_grad.cpu())


if __name__ == "__main__":
    run_tests()
