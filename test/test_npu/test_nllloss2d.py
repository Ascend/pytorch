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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from torch.autograd import Variable

class TestNllloss2d(TestCase):
    def cpu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output
 
    def npu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output
    
    def test_nll_loss2d_mean(self, device):
        m = torch.nn.LogSoftmax(dim=1)
        N, C = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, C, (3, 3))
        data = m(conv(torch.randn(N, 16, 10, 10)))
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_exec(data, target, "mean")
        npu_output = self.npu_op_exec(data_npu, target_npu, "mean")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_none(self, device):
        m = torch.nn.LogSoftmax(dim=1)
        N, C = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, C, (3, 3))
        data = m(conv(torch.randn(N, 16, 10, 10)))
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_exec(data, target, "none")
        npu_output = self.npu_op_exec(data_npu, target_npu, "none")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_sum(self, device):
        m = torch.nn.LogSoftmax(dim=1)
        N, C = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, C, (3, 3))
        data = m(conv(torch.randn(N, 16, 10, 10)))
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_exec(data, target, "sum")
        npu_output = self.npu_op_exec(data_npu, target_npu, "sum")

        self.assertRtolEqual(cpu_output, npu_output)
    
    def cpu_op_grad_exec(self, data, target, reduction):
        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook
        data = Variable(data, requires_grad=True)
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        torch_result = loss(data, target)

        if reduction == "none":
            torch_result = torch_result.sum()

        data.register_hook(save_grad('x'))
        torch_result.backward()
        output = grads['x'].to("cpu").numpy()
        return output
 
    def npu_op_grad_exec(self, data, target, reduction):
        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook
        data = Variable(data, requires_grad=True)
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        torch_result = loss(data, target)

        if reduction == "none":
            torch_result = torch_result.sum()

        data.register_hook(save_grad('x'))
        torch_result.backward()
        output = grads['x'].to("cpu").numpy()
        return output
    
    def test_nll_loss2d_grad_mean(self, device):
        m = torch.nn.LogSoftmax(dim=1)
        N, C = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, C, (3, 3))
        data = m(conv(torch.randn(N, 16, 10, 10)))
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_grad_exec(data, target, "mean")
        npu_output = self.npu_op_grad_exec(data_npu, target_npu, "mean")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_grad_none(self, device):
        m = torch.nn.LogSoftmax(dim=1)
        N, C = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, C, (3, 3))
        data = m(conv(torch.randn(N, 16, 10, 10)))
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_grad_exec(data, target, "none")
        npu_output = self.npu_op_grad_exec(data_npu, target_npu, "none")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_grad_sum(self, device):
        m = torch.nn.LogSoftmax(dim=1)
        N, C = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, C, (3, 3))
        data = m(conv(torch.randn(N, 16, 10, 10)))
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_grad_exec(data, target, "sum")
        npu_output = self.npu_op_grad_exec(data_npu, target_npu, "sum")

        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestNllloss2d, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
