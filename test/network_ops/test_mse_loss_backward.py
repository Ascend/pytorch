# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
from torch.autograd import Variable
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestMseLossGrad(TestCase):

    def generate_mse_grad_inputs(self, min_ref, max_ref, shape, dtype):
        input1 = np.random.uniform(min_ref, max_ref, shape).astype(dtype)
        input2 = np.random.uniform(min_ref, max_ref, shape).astype(dtype)

        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def cpu_op_exec_default(self, input1, input2):
        grads = {}

        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook

        input1 = Variable(input1, requires_grad=True)

        output = torch.nn.functional.mse_loss(input1, input2)
        dout_result = 1.23 * output
        torch_result = dout_result.sum()

        input1.register_hook(save_grad('x'))
        torch_result.backward()

        output = grads['x'].numpy()
        return output

    def npu_op_exec_default(self, input1, input2):
        grads = {}

        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook

        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input1 = Variable(input1, requires_grad=True)

        output = torch.nn.functional.mse_loss(input1, input2)
        dout_result = 1.23 * output
        torch_result = dout_result.sum()

        input1.register_hook(save_grad('x'))
        torch_result.backward()

        output = grads['x'].to("cpu").numpy()
        return output

    def cpu_op_exec(self, input1, input2, reduction):
        grads = {}

        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook

        input1 = Variable(input1, requires_grad=True)

        output = torch.nn.functional.mse_loss(input1, input2, reduction=reduction)
        dout_result = 1.23 * output
        torch_result = dout_result.sum()

        input1.register_hook(save_grad('x'))
        torch_result.backward()

        output = grads['x'].numpy()
        return output

    def npu_op_exec(self, input1, input2, reduction):
        grads = {}

        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook

        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input1 = Variable(input1, requires_grad=True)

        output = torch.nn.functional.mse_loss(input1, input2, reduction=reduction)
        dout_result = 1.23 * output
        torch_result = dout_result.sum()

        input1.register_hook(save_grad('x'))
        torch_result.backward()

        output = grads['x'].to("cpu").detach().numpy()
        return output

    def test_mse_loss_grad_float32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_mse_grad_inputs(0, 100, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_default(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_mse_loss_grad_float32_mean(self, device="npu"):
        npu_input1, npu_input2 = self.generate_mse_grad_inputs(0, 100, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "mean")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "mean")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_mse_loss_grad_float32_none(self, device="npu"):
        npu_input1, npu_input2 = self.generate_mse_grad_inputs(0, 100, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "none")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "none")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_mse_loss_grad_float32_sum(self, device="npu"):
        npu_input1, npu_input2 = self.generate_mse_grad_inputs(0, 100, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "sum")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "sum")
        self.assertRtolEqual(cpu_output, npu_output)

    def test_mse_loss_grad_shape_0(self, device="npu"):
        npu_input1, npu_input2 = self.generate_mse_grad_inputs(0, 100, (0, 4), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, "mean")
        npu_output = self.npu_op_exec(npu_input1, npu_input2, "mean")
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
