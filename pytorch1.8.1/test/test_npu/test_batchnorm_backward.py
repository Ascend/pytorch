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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestBn2d(TestCase):
    def cpu_op_exec(self,input1, dim, weight, bias, weight_grad=True, bias_grad=True):
        input1.requires_grad_(True)
        m = torch.nn.BatchNorm2d(dim)
        m.weight.data = weight
        if not weight_grad:
            m.weight.requires_grad = False
        m.bias.data = bias
        if  not bias_grad:
            m.bias.requires_grad = False
        input_cpu = m(input1)
        input_cpu = input_cpu.detach().numpy()
        w = torch.ones_like(input1)
        tmp = m(input1)
        tmp.backward(w)
        input_grad = input1.grad.detach().numpy()
        weight_grad = m.weight.grad
        bias_grad = m.bias.grad
        return input_grad, weight_grad, bias_grad, input_cpu

    def npu_op_exec_new(self,input1, dim, weight, bias, weight_grad=True, bias_grad=True):
        w = torch.ones_like(input1)
        m = torch.nn.BatchNorm2d(dim)
        m.weight.data = weight
        if not weight_grad:
            m.weight.requires_grad = False
        m.bias.data = bias
        if  not bias_grad:
            m.bias.requires_grad = False
        m = m.to("npu")
        input_npu = m(input1)
        input_npu = input_npu.to("cpu")
        input_npu = input_npu.detach().numpy()
        input1.requires_grad_(True)
        tmp = m(input1)
        tmp.backward(w)
        input_grad = input1.grad.to("cpu").detach().numpy()
        weight_grad = m.weight.grad
        bias_grad = m.bias.grad
        return input_grad, weight_grad, bias_grad, input_npu

    def test_batchnorm_shape_format(self, device):
        shape_format = [
                [[np.float16, 0, [256, 32, 112, 112]]],
                [[np.float16, 3, [256, 672, 7, 7]]],
                [[np.float16, 4, [256, 288, 14, 14]]],
                [[np.float16, 29, [1024, 58, 28, 28]]],
                [[np.float16, 0, [1024, 116, 14, 14]]],
                [[np.float16, 3, [1024, 24, 112, 112]]],
                [[np.float16, 4, [1024, 58, 56, 56]]],
                [[np.float16, 29, [1024, 1024, 7, 7]]],
                [[np.float16, 0, [1024, 24, 28, 28]]],
                [[np.float16, 3, [1024, 116, 28, 28]]],
                [[np.float16, 4, [1024, 232, 7, 7]]],
                [[np.float16, 29, [1024, 232, 14, 14]]],
                [[np.float32, 0, [256, 32, 112, 112]]],
                [[np.float32, 3, [256, 672, 7, 7]]],
                [[np.float32, 4, [256, 288, 14, 14]]],
                [[np.float32, 29, [1024, 58, 28, 28]]],
                [[np.float32, 0, [1024, 116, 14, 14]]],
                [[np.float32, 3, [1024, 24, 112, 112]]],
                [[np.float32, 4, [1024, 58, 56, 56]]],
                [[np.float32, 29, [1024, 1024, 7, 7]]],
                [[np.float32, 0, [1024, 24, 28, 28]]],
                [[np.float32, 3, [1024, 116, 28, 28]]],
                [[np.float32, 4, [1024, 232, 7, 7]]],
                [[np.float32, 29, [1024, 232, 14, 14]]],
         ]
        wieght_bias_grads = [[True, True],[True, False],[False, True],[False, False],]
        for wieght_bias_grad in wieght_bias_grads:
            for item in shape_format:
                cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
                weight_bias = item[0][:]
                weight_bias[0] = np.float32
                weight_bias[2] = item[0][2][1]
                cpu_weight, npu_weight = create_common_tensor(weight_bias, 0, 100)
                cpu_bias, npu_bias = create_common_tensor(weight_bias, 0, 100)
                if cpu_input1.dtype == torch.float16:
                    cpu_input1 = cpu_input1.to(torch.float32)
                input_grad_cpu, weight_grad_cpu, bias_grad_cpu, input_cpu = \
                    self.cpu_op_exec(cpu_input1, item[0][2][1], cpu_weight, cpu_bias, *wieght_bias_grad)
                input_grad_npu, weight_grad_npu, bias_grad_npu, input_npu=  \
                    self.npu_op_exec_new(npu_input1, item[0][2][1], npu_weight, npu_bias, *wieght_bias_grad)
                input_grad_cpu = input_grad_cpu.astype(input_grad_npu.dtype)
                self.assertRtolEqual(input_grad_cpu, input_grad_npu)

                if wieght_bias_grad[0]:
                    weight_grad_cpu = weight_grad_cpu.detach().numpy()
                    weight_grad_npu = weight_grad_npu.to("cpu").detach().numpy()
                    weight_grad_cpu = weight_grad_cpu.astype(weight_grad_npu.dtype)
                    self.assertRtolEqual(weight_grad_cpu, weight_grad_npu)
                else:
                    assert (weight_grad_cpu is None) and (weight_grad_npu is None)
                if wieght_bias_grad[1]:
                    bias_grad_cpu = bias_grad_cpu.detach().numpy()
                    bias_grad_npu = bias_grad_npu.to("cpu").detach().numpy()
                    bias_grad_cpu = bias_grad_cpu.astype(bias_grad_npu.dtype)
                    self.assertRtolEqual(bias_grad_cpu, bias_grad_npu)
                else:
                    assert (bias_grad_cpu is None) and (bias_grad_npu is None)
                input_cpu = input_cpu.astype(input_npu.dtype)
                self.assertRtolEqual(input_cpu, input_npu)


instantiate_device_type_tests(TestBn2d, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
