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

cpu_input_grad = None
npu_input_grad = None


def cpu_input_grad_hook(grad):
    global cpu_input_grad
    cpu_input_grad = grad


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.cpu()


class TestHardSigmoidBackward(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_grad = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input_grad = torch.from_numpy(input_grad)
        input_x = torch.from_numpy(input_x)
        return input_grad, input_x

    def cpu_op_exec(self, input_x, input_grad):
        input_x.requires_grad_(True)
        input_x.register_hook(cpu_input_grad_hook)
        h = torch.nn.Hardsigmoid()
        output = h(input_x)
        output.backward(input_grad)

    def npu_op_exec(self, input_x, input_grad):
        input_x = input_x.to("npu")
        input_grad = input_grad.to("npu")
        input_x.requires_grad_(True)
        input_x.register_hook(npu_input_grad_hook)
        h = torch.nn.Hardsigmoid()
        output = h(input_x)
        output.backward(input_grad)

    def test_hardsigmoidbackward_6_6_float32(self, device="npu"):
        input_grad, input_x = self.generate_data(-6, 6, (6, 6), np.float32)
        self.cpu_op_exec(input_x, input_grad)
        self.npu_op_exec(input_x, input_grad)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_hardsigmoidbackward_10_10_float32(self, device="npu"):
        input_grad, input_x = self.generate_data(-6, 6, (10, 10), np.float32)
        self.cpu_op_exec(input_x, input_grad)
        self.npu_op_exec(input_x, input_grad)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_hardsigmoidbackward_100_100_float32(self, device="npu"):
        input_grad, input_x = self.generate_data(-6, 6, (100, 100), np.float32)
        self.cpu_op_exec(input_x, input_grad)
        self.npu_op_exec(input_x, input_grad)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_hardsigmoidbackward_10_10_10_10_float32(self, device="npu"):
        input_grad, input_x = self.generate_data(-6, 6, (10, 10, 10, 10), np.float32)
        self.cpu_op_exec(input_x, input_grad)
        self.npu_op_exec(input_x, input_grad)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_hardsigmoidbackward_6_6_float16(self, device="npu"):
        input_grad1, input_x1 = self.generate_data(-6, 6, (6, 6), np.float16)
        input_grad1 = input_grad1.to(torch.float32)
        input_x1 = input_x1.to(torch.float32)
        self.cpu_op_exec(input_x1, input_grad1)
        self.npu_op_exec(input_x1, input_grad1)
        self.assertRtolEqual(cpu_input_grad.detach().numpy().astype(npu_input_grad.detach().numpy().dtype),
                             npu_input_grad.detach().numpy())

    def test_hardsigmoidbackward_10_10_float16(self, device="npu"):
        input_grad1, input_x1 = self.generate_data(-6, 6, (10, 10), np.float16)
        input_grad1 = input_grad1.to(torch.float32)
        input_x1 = input_x1.to(torch.float32)
        self.cpu_op_exec(input_x1, input_grad1)
        self.npu_op_exec(input_x1, input_grad1)
        self.assertRtolEqual(cpu_input_grad.detach().numpy().astype(npu_input_grad.detach().numpy().dtype),
                             npu_input_grad.detach().numpy())

    def test_hardsigmoidbackward_100_100_float16(self, device="npu"):
        input_grad1, input_x1 = self.generate_data(-6, 6, (100, 100), np.float16)
        input_grad1 = input_grad1.to(torch.float32)
        input_x1 = input_x1.to(torch.float32)
        self.cpu_op_exec(input_x1, input_grad1)
        self.npu_op_exec(input_x1, input_grad1)
        self.assertRtolEqual(cpu_input_grad.detach().numpy().astype(npu_input_grad.detach().numpy().dtype),
                             npu_input_grad.detach().numpy())

    def test_hardsigmoidbackward_10_10_10_10_float16(self, device="npu"):
        input_grad1, input_x1 = self.generate_data(-6, 6, (10, 10, 10, 10), np.float16)
        input_grad1 = input_grad1.to(torch.float32)
        input_x1 = input_x1.to(torch.float32)
        self.cpu_op_exec(input_x1, input_grad1)
        self.npu_op_exec(input_x1, input_grad1)
        self.assertRtolEqual(cpu_input_grad.detach().numpy().astype(npu_input_grad.detach().numpy().dtype),
                             npu_input_grad.detach().numpy())


if __name__ == '__main__':
    run_tests()
