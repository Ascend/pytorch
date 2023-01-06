# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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


import copy

import numpy as np
import torch
import torch.nn as nn

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class TestConv2dBackwardBackward(TestCase):
    def test_conv2d_backward_backward(self):
        x_cpu = torch.randn(16, 3, 128, 128)
        x_npu = copy.deepcopy(x_cpu).npu()

        x_cpu.requires_grad = True
        x_npu.requires_grad = True

        model_cpu = Discriminator()
        model_npu = copy.deepcopy(model_cpu).npu()

        # forward
        y_cpu = model_cpu(x_cpu)
        y_npu = model_npu(x_npu)
        torch.npu.synchronize()

        # backward
        grad_outputs_cpu = torch.ones(y_cpu.size())
        grad_outputs_npu = copy.deepcopy(grad_outputs_cpu).npu()
        loss_cpu = torch.autograd.grad(
            outputs=y_cpu,
            inputs=x_cpu,
            grad_outputs=grad_outputs_cpu,
            retain_graph=True,
            create_graph=True,
            only_inputs=True)[0]
        loss_npu = torch.autograd.grad(
            outputs=y_npu,
            inputs=x_npu,
            grad_outputs=grad_outputs_npu,
            retain_graph=True,
            create_graph=True,
            only_inputs=True)[0]
        torch.npu.synchronize()

        # backward-backward
        loss_cpu.backward(torch.ones_like(x_cpu))
        loss_npu.backward(torch.ones_like(x_npu))
        self.assertRtolEqual(x_cpu.grad.numpy(), x_npu.grad.cpu().numpy())


if __name__ == "__main__":
    run_tests()
