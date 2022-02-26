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
import torch_npu
import torch.nn.functional as F
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests


class TestKlDiv(TestCase):
    def cpu_op_exec(self, input1, input2, reduction):
        output = torch.kl_div(input1, input2, reduction=reduction)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, reduction):
        output = torch.kl_div(input1, input2, reduction=reduction)
        output = output.cpu()
        output = output.numpy()
        return output
    
    def test_kl_div_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[torch.float32, 0, (192, 8)], [torch.float32, 0, (192, 8)], 1],
            [[torch.float32, 0, (192, 500)], [torch.float32, 0, (192, 500)], 1],
            [[torch.float32, 0, (2, 3)], [torch.float32, 0, (2, 3)], 2],
            [[torch.float32, 0, (4, 5)], [torch.float32, 0, (4, 5)], 2],
            [[torch.float32, 0, (2, 3, 3)], [torch.float32, 0, (2, 3, 3)], 2],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input = F.log_softmax(x, dim = 0)
            cpu_target = F.softmax(y, dim = 0)
            npu_input = cpu_input.npu()
            npu_target = cpu_target.npu()
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input, cpu_target, reduction)
            npu_output = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_kl_div_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[torch.float16, 0, (192, 8)], [torch.float16, 0, (192, 8)], 1],
            [[torch.float16, 0, (192, 200)], [torch.float16, 0, (192, 200)], 1],
            [[torch.float16, 0, (2, 3)], [torch.float16, 0, (2, 3)], 2],
            [[torch.float16, 0, (4, 5)], [torch.float16, 0, (4, 5)], 2],
            [[torch.float16, 0, (2, 3, 3)], [torch.float16, 0, (2, 3, 3)], 2],
        ]
        for item in shape_format:
            x = torch.randn(item[0][2])
            y = torch.randn(item[1][2])
            cpu_input = F.log_softmax(x, dim = 0).to(item[0][0])
            cpu_target = F.softmax(y, dim = 0).to(item[0][0])
            npu_input = cpu_input.npu()
            npu_target = cpu_target.npu()
            reduction = item[2]
            cpu_output = self.cpu_op_exec(cpu_input.to(torch.float32), cpu_target.to(torch.float32), reduction)
            npu_output = self.npu_op_exec(npu_input, npu_target, reduction)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)


if __name__ == "__main__":
    run_tests()
