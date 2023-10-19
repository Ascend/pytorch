# Copyright (c) 2023, Huawei Technologies.
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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRreluWithNoise(TestCase):

    def op_exec_cpu(self, input1, input2, lower, upper, train):
        input1.requires_grad = True

        cpu_output = torch._C._nn.rrelu_with_noise(input1, input2, lower, upper, train)
        tmp = torch.ones_like(cpu_output)
        cpu_output.backward(tmp)

        return cpu_output.detach().numpy(), input1.grad.numpy()

    def op_exec_npu(self, input1, input2, lower, upper, train):
        input1.requires_grad = True

        npu_output = torch._C._nn.rrelu_with_noise(input1, input2, lower, upper, train)
        tmp = torch.ones_like(npu_output)
        npu_output.backward(tmp)
        npu_output = npu_output.cpu()
        return npu_output.detach().cpu().numpy(), input1.grad.cpu().numpy()

    def check_forward_backward(self, shape_format, lower, upper, train):
        for item in shape_format:
            input1_cpu, input1_npu = create_common_tensor(item[0], -10, 10)
            if input1_cpu.dtype == torch.float16:
                input1_cpu = input1_cpu.to(torch.float32)
            input2_cpu, input2_npu = create_common_tensor(item[1], -10, 10)
            if input2_cpu.dtype == torch.float16:
                input2_cpu = input2_cpu.to(torch.float32)
            cpu_output, cpu_input1_grad = self.op_exec_cpu(input1_cpu, input2_cpu, lower, upper, train)
            npu_output, npu_input1_grad = self.op_exec_npu(input1_npu, input2_npu, lower, upper, train)

            self.assertRtolEqual(cpu_output.astype(npu_output.dtype), npu_output)
            self.assertRtolEqual(cpu_input1_grad.astype(npu_input1_grad.dtype), npu_input1_grad)

    def test_rrelu(self):
        shape_format = [
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [5, 7, 10]]],
            [[np.float32, 2, [5, 6]], [np.float32, 2, [5, 6]]],
        ]
        self.check_forward_backward(shape_format, lower=0.1, upper=0.3, train=False)
        self.check_forward_backward(shape_format, lower=0.1, upper=0.1, train=True)
        self.check_forward_backward(shape_format, lower=0.1, upper=0.1, train=False)


if __name__ == "__main__":
    run_tests()
