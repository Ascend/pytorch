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


import copy
import torch
import torch.nn as nn
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


def generate_data(min1, max1, shape, dtype):
    input1 = np.random.uniform(min1, max1, shape).astype(dtype)
    # modify from numpy.ndarray to torch.tensor
    output = torch.from_numpy(input1)
    # generate target: target.size == input1.size
    label = torch.randint(shape[1], size=(shape[0],), dtype=torch.long)
    target = torch.zeros(shape[0], shape[1])
    target[range(target.shape[0]), label] = 1
    target = target.to(output.dtype)
    return output, target

class TestBinaryCrossEntropyWithLogitsBackward(TestCase):

    def cpu_op_exec(self, input1, target):
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target)
        input_cpu = output.detach().numpy()
        output.backward()
        res = input1.grad
        res = res.numpy()
        return input_cpu, res

    def npu_op_exec(self, input1, target):
        target = target.to("npu")
        input1 = input1.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target)
        input_npu = output.cpu()
        input_npu = input_npu.detach().numpy()
        output.backward()
        res = input1.grad.cpu()
        res = res.numpy()
        return input_npu, res

    def cpu_op_exec_pos_weight_int(self, input1, target, weight=None, pos_weight=None):
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target, weight=weight,
                                                                      pos_weight=pos_weight)
        input_cpu = output.detach().numpy()
        output.backward()
        res = input1.grad.numpy()
        return input_cpu, res

    def npu_op_exec_pos_weight_int(self, input1, target, weight=None, pos_weight=None):
        input1.requires_grad_(True)
        output = torch.nn.functional.binary_cross_entropy_with_logits(input1, target, weight=weight,
                                                                      pos_weight=pos_weight)
        input_npu = output.cpu()
        input_npu = input_npu.detach().numpy()
        output.backward()
        res = input1.grad.cpu().numpy()
        return input_npu, res

    def test_binary_cross_entropy_with_logits_backward_fp32(self):
        npu_input1, npu_target = generate_data(0, 100, (5, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_target = copy.deepcopy(npu_target)
        cpu_output, cpu_grad_output = self.cpu_op_exec(cpu_input1, cpu_target)
        npu_output, npu_grad_output = self.npu_op_exec(npu_input1, npu_target)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)

    def test_binary_cross_entropy_with_logits_backward_fp16(self):
        npu_input1, npu_target = generate_data(0, 100, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_target = copy.deepcopy(npu_target)
        cpu_input1 = cpu_input1.to(torch.float32)
        cpu_target = cpu_target.to(torch.float32)
        cpu_output, cpu_grad_output = self.cpu_op_exec(cpu_input1, cpu_target)
        npu_output, npu_grad_output = self.npu_op_exec(npu_input1, npu_target)
        cpu_output = cpu_output.astype(npu_output.dtype)
        cpu_grad_output = cpu_grad_output.astype(npu_grad_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)

    def test_binary_cross_entropy_with_logits_backward_fp32_pos_weight_int(self):
        cpu_input1, cpu_target = generate_data(0, 100, (5, 3), np.float32)
        npu_input1 = cpu_input1.npu()
        npu_target = cpu_target.npu()
        shape_format = [np.int64, -1, (5, 3)]
        weight_cpu, weight_npu = create_common_tensor(shape_format, 0, 100)
        pos_weight_cpu, pos_weight_npu = create_common_tensor(shape_format, 0, 100)
        cpu_output, cpu_grad_output = self.cpu_op_exec_pos_weight_int(cpu_input1, cpu_target,
                                                                      weight=weight_cpu, pos_weight=pos_weight_cpu)
        npu_output, npu_grad_output = self.npu_op_exec_pos_weight_int(npu_input1, npu_target,
                                                                      weight=weight_npu, pos_weight=pos_weight_npu)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)

    def test_binary_cross_entropy_with_logits_backward_fp16_pos_weight_int(self):
        cpu_input1, cpu_target = generate_data(0, 100, (5, 3), np.float16)
        npu_input1 = cpu_input1.npu()
        npu_target = cpu_target.npu()
        cpu_input1 = cpu_input1.to(torch.float32)
        cpu_target = cpu_target.to(torch.float32)
        shape_format = [np.int64, -1, (5, 3)]
        weight_cpu, weight_npu = create_common_tensor(shape_format, 0, 100)
        pos_weight_cpu, pos_weight_npu = create_common_tensor(shape_format, 0, 100)
        cpu_output, cpu_grad_output = self.cpu_op_exec_pos_weight_int(cpu_input1, cpu_target,
                                                                      weight=weight_cpu, pos_weight=pos_weight_cpu)
        npu_output, npu_grad_output = self.npu_op_exec_pos_weight_int(npu_input1, npu_target,
                                                                      weight=weight_npu, pos_weight=pos_weight_npu)
        cpu_output = cpu_output.astype(npu_output.dtype)
        cpu_grad_output = cpu_grad_output.astype(npu_grad_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_grad_output, npu_grad_output)


if __name__ == "__main__":
    run_tests()
