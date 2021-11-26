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
from graph_utils import RunFuncInGraphMode

class TestNllloss(TestCase):
    def cpu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output
 
    def npu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output
    
    @RunFuncInGraphMode
    def test_nll_loss_mean(self, device):
        N, C = 5, 4
        data = torch.randn(N, C)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_exec(data, target, "mean")
        npu_output = self.npu_op_exec(data_npu, target_npu, "mean")

        self.assertRtolEqual(cpu_output, npu_output)
    
    @RunFuncInGraphMode
    def test_nll_loss_none(self, device):
        N, C = 5, 4
        data = torch.randn(N, C)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_exec(data, target, "none")
        npu_output = self.npu_op_exec(data_npu, target_npu, "none")

        self.assertRtolEqual(cpu_output, npu_output)
    
    @RunFuncInGraphMode
    def test_nll_loss_sum(self, device):
        N, C = 5, 4
        data = torch.randn(N, C)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

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
        loss = torch.nn.NLLLoss(reduction=reduction)
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
        loss = torch.nn.NLLLoss(reduction=reduction)
        torch_result = loss(data, target)

        if reduction == "none":
            torch_result = torch_result.sum()

        data.register_hook(save_grad('x'))
        torch_result.backward()
        output = grads['x'].to("cpu").numpy()
        return output
    
    @RunFuncInGraphMode
    def test_nll_loss_grad_mean(self, device):
        N, C = 5, 4
        data = torch.randn(N, C)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_grad_exec(data, target, "mean")
        npu_output = self.npu_op_grad_exec(data_npu, target_npu, "mean")

        self.assertRtolEqual(cpu_output, npu_output)
    
    @RunFuncInGraphMode
    def test_nll_loss_grad_none(self, device):
        N, C = 5, 4
        data = torch.randn(N, C)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_grad_exec(data, target, "none")
        npu_output = self.npu_op_grad_exec(data_npu, target_npu, "none")

        self.assertRtolEqual(cpu_output, npu_output)
    
    @RunFuncInGraphMode
    def test_nll_loss_grad_sum(self, device):
        N, C = 5, 4
        data = torch.randn(N, C)
        target = torch.empty(N, dtype=torch.long).random_(0, C)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)
        
        cpu_output = self.cpu_op_grad_exec(data, target, "sum")
        npu_output = self.npu_op_grad_exec(data_npu, target_npu, "sum")

        self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec_new(self, input1, target, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100 # 默认值
        output = torch.nn.functional.nll_loss(input1, target, reduction=reduction, ignore_index=ignore_index)
        output = output.numpy()
        return output

    def npu_op_exec_new(self, input1, target, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100 # 默认值
        target = target.to(torch.int32)
        target = target.to("npu")
        output = torch.nn.functional.nll_loss(input1, target, reduction=reduction, ignore_index=ignore_index)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    @RunFuncInGraphMode
    def test_nllloss_shape_format_fp32(self, device):
        # 当前仅支持设置正数, 若np.sum(ignore_index == np_target) == 0,则ignore_index设置任意数值不影响
        ignore_index = 1 
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float32, 0, [256, 100]], [np.int32, 0, [256]], reduction, None],
                [[np.float32, 3, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float32, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float32, 3, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float32, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, None],
                ]
            for item in shape_format:
                np_target = np.random.randint(0, item[0][2][1], (item[1][2])).astype(np.long)
                target = torch.from_numpy(np_target)
                cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
                cpu_output = self.cpu_op_exec_new(cpu_input1, target, item[2], item[3])
                npu_output = self.npu_op_exec_new(npu_input1, target, item[2], item[3])
                self.assertRtolEqual(cpu_output, npu_output)

    @RunFuncInGraphMode
    def test_nllloss_shape_format_fp16(self, device):
        # 当前仅支持设置正数, 若np.sum(ignore_index == np_target) == 0,则ignore_index设置任意数值不影响
        ignore_index = 1
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float16, 0, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float16, 3, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float16, 3, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, None],
                ]
            for item in shape_format:
                np_target = np.random.uniform(0, item[0][2][1], (item[1][2])).astype(np.long)
                target = torch.from_numpy(np_target)
                cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_output = self.cpu_op_exec_new(cpu_input1, target, item[2], item[3])
                npu_output = self.npu_op_exec_new(npu_input1, target, item[2], item[3])
                cpu_output = cpu_output.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec_mid(self, input1, target, weight, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100 # 默认值
        weight = weight + 0
        output = torch.nn.functional.nll_loss(input1, target, weight, reduction=reduction, ignore_index=ignore_index)
        output = output.numpy()
        return output

    def npu_op_exec_mid(self, input1, target, weight, reduction, ignore_index):
        if not ignore_index:
            ignore_index = -100 # 默认值
        target = target.to(torch.int32)
        target = target.to("npu")
        weight = weight + 0
        output = torch.nn.functional.nll_loss(input1, target, weight, reduction=reduction, ignore_index=ignore_index)
        output = output.to("cpu")
        output = output.numpy()
        return output


    @RunFuncInGraphMode
    def test_nllloss_graph_mid(self, device):
        # 当前仅支持设置正数, 若np.sum(ignore_index == np_target) == 0,则ignore_index设置任意数值不影响
        ignore_index = 1
        for reduction in ['mean', 'none', 'sum']:
            shape_format = [
                [[np.float16, 0, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float16, 3, [256, 100]], [np.int32, 0, [256]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float16, 3, [4800, 3003]], [np.int32, 0, [4800]], reduction, ignore_index],
                [[np.float16, 0, [4800, 3003]], [np.int32, 0, [4800]], reduction, None],
                ]
            for item in shape_format:
                np_target = np.random.uniform(0, item[0][2][1], (item[1][2])).astype(np.long)
                target = torch.from_numpy(np_target)
                cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
                cpu_input1 = cpu_input1.to(torch.float32)
                weight_cpu = torch.ones(item[0][2][1], dtype=cpu_input1.dtype)
                weight_npu = torch.ones(item[0][2][1], dtype=npu_input1.dtype).to("npu")
                cpu_output = self.cpu_op_exec_mid(cpu_input1, target, weight_cpu, item[2], item[3])
                npu_output = self.npu_op_exec_mid(npu_input1, target, weight_npu, item[2], item[3])
                cpu_output = cpu_output.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)
instantiate_device_type_tests(TestNllloss, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
