# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from torch._C import _infer_size

class TestBinaryCrossEntropyExt(TestCase):
    def generate_data(self, min_val, max_val, shape, dtype):
        x = np.random.uniform(min_val, max_val, shape).astype(dtype)
        x = torch.from_numpy(x)
        return x

    def cpu_op_exec(self, input1, target, weight, reduction="mean"):
        float16flag = False
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            target = target.to(torch.float32)
            float16flag = True
            if weight is not None:
                weight = weight.to(torch.float32)
        cpu_output = torch.nn.functional.binary_cross_entropy(input1, target, weight=weight, size_average=None, reduce=None, reduction=reduction)
        if float16flag:
            cpu_output = cpu_output.to(torch.float16)
        cpu_output = cpu_output.detach().numpy()
        return cpu_output

    #.out接口如下,reduction 为int类型，非str类型，weight需要进行expand，另cpu output可以广播，npu不支持。
    # 当reduction为mean，sum时，npu output shape为[],cpu 为input shape。
    def cpu_op_exec_out(self, intput1,target, weight, output, reduction = "mean"):  
        if reduction == "none":
            reductions = 0
        elif reduction == "mean":
            reductions = 1
        elif reduction == "sum":
            reductions = 2
        float16flag = False
        if intput1.dtype == torch.float16:
            intput1 = intput1.to(torch.float32)
            target = target.to(torch.float32)
            output = output.to(torch.float32)
            float16flag = True
        if weight is not None:
            new_size = _infer_size(target.size(), weight.size())
            weight = weight.expand(new_size)
            if float16flag:
                weight = weight.to(torch.float32)
        cpu_output = torch._C._nn.binary_cross_entropy(intput1, target, weight = weight, out = output, reduction = reductions)
        if float16flag:
            cpu_output = cpu_output.to(torch.float16)
        cpu_output = cpu_output.detach().numpy()
        return cpu_output

    def npu_op_exec(self, input1, target, weight, reduction="mean"):
        input1 = input1.npu()
        target = target.npu()
        if weight is not None:
            weight = weight.npu()
        npu_output = torch.nn.functional.binary_cross_entropy(input1, target, weight=weight, size_average=None, reduce=None, reduction=reduction)
        npu_output = npu_output.cpu()
        npu_output = npu_output.detach().numpy()
        return npu_output

    def npu_op_exec_out(self, input1,target, weight, output, reduction = "mean"):
        input1 = input1.npu()
        target = target.npu()
        output = output.npu()
        if reduction == "none":
           reductions = 0
        elif reduction == "mean":
            reductions = 1
        elif reduction == "sum":
            reductions = 2
        if weight is not None:
            weight = weight.npu()
            new_size = _infer_size(target.size(), weight.size())
            weight = weight.expand(new_size)
        npu_output = torch._C._nn.binary_cross_entropy(input1,target,weight = weight, out = output, reduction = reductions)
        npu_output = npu_output.cpu()
        npu_output = npu_output.detach().numpy()
        return npu_output

    def test_binary_cross_entropy_float16(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float16, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 2, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            weight = None
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_target, weight, reduction=item[2])
            npu_output = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_out_float16(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float16, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 2, item[1], item[0])
            if item[2] == 'mean' or item[2] == 'sum':
                output = self.generate_data(0, 1, [], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            else:
                output = self.generate_data(0, 1, item[1], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            weight = None
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_target, weight, cpu_output, reduction=item[2])
            npu_output = self.npu_op_exec_out(input1, target, weight, output,reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_float32(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float32, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 2, item[1], item[0]).int().to(torch.float32)
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            weight = None
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_target, weight, reduction=item[2])
            npu_output = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_out_float32(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float32, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            if item[2] == 'mean' or item[2] == 'sum':
                output = self.generate_data(0, 1, [], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            else:
                output = self.generate_data(0, 1, item[1], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            target = self.generate_data(0, 2, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_output = copy.deepcopy(output)
            weight = None
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_target, weight, cpu_output,reduction = item[2])
            npu_output = self.npu_op_exec_out(input1, target, weight, output, reduction = item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_with_weight_float16(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none","mean", "sum"]
        shape_format = [
            [np.float16, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 2, item[1], item[0])
            weight = self.generate_data(0, 1, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_weight = copy.deepcopy(weight)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_target, cpu_weight, reduction=item[2])
            npu_output = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_out_with_weight_float16(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float16, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 2, item[1], item[0])
            weight = self.generate_data(0, 1, item[1], item[0])
            if item[2] == 'mean' or item[2] == 'sum':
                output = self.generate_data(0, 1, [], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            else:
                output = self.generate_data(0, 1, item[1], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_weight = copy.deepcopy(weight)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_target,
                                                        cpu_weight, cpu_output,reduction=item[2])
            npu_output = self.npu_op_exec_out(input1, target, weight, output, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_with_weight_float32(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float32, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 1, item[1], item[0])
            weight = self.generate_data(0, 1, item[1], item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_weight = copy.deepcopy(weight)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_target, cpu_weight, reduction=item[2])
            npu_output = self.npu_op_exec(input1, target, weight, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_binary_cross_entropy_out_with_weight_float32(self, device):
        shape_list = [[1024], [32, 1024], [32, 8, 1024]]
        reduction_list = ["none", "mean", "sum"]
        shape_format = [
            [np.float32, i, j] for i in shape_list for j in reduction_list
        ]
        for item in shape_format:
            input1 = self.generate_data(0, 1, item[1], item[0])
            target = self.generate_data(0, 2, item[1], item[0])
            weight = self.generate_data(0, 1, item[1], item[0])
            if item[2] == 'mean' or item[2] == 'sum':
                output = self.generate_data(0, 1, [], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            else:
                output = self.generate_data(0, 1, item[1], item[0])
                cpu_output = self.generate_data(0, 1, item[1],item[0])
            cpu_input1 = copy.deepcopy(input1)
            cpu_target = copy.deepcopy(target)
            cpu_weight = copy.deepcopy(weight)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_target,
                                                        cpu_weight, cpu_output,reduction=item[2])
            npu_output = self.npu_op_exec_out(input1, target, weight, output, reduction=item[2])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestBinaryCrossEntropyExt, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()
