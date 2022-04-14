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

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMaxPool2dBackward(TestCase):
    def cpu_op_exec(self, inputCpu, kernel_size, stride, padding):
        inputCpu.requires_grad = True
        dataCpu, argMaxCpu = F.max_pool2d_with_indices(inputCpu, kernel_size=kernel_size, stride=stride,
                                                       padding=padding)
        z1 = torch.sum(dataCpu)
        z1.backward()
        cpu_grad = inputCpu.grad
        output1 = dataCpu.detach()
        output1 = output1
        return output1, cpu_grad

    def npu_op_exec(self, inputNpu, kernel_size, stride, padding):
        inputNpu.requires_grad = True
        dataNpu, argMaxNpu = F.max_pool2d_with_indices(inputNpu, kernel_size=kernel_size, stride=stride,
                                                       padding=padding)
        z2 = torch.sum(dataNpu)
        z2.backward()
        npu_grad = inputNpu.grad
        npu_grad = npu_grad.to("cpu")
        output1 = dataNpu.to("cpu").detach()
        return output1, npu_grad

    def test_max_pool2d_backward_shape_format(self, device):
        shape_format = [
            [[np.float16, 3, [256, 64, 112, 112]], [3, 3], [2, 2], 1],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, item[1], item[2], item[3])
            npu_output, npu_grad = self.npu_op_exec(npu_input, item[1], item[2], item[3])
            cpu_output = cpu_output.to(npu_output.dtype)
            cpu_grad = cpu_grad.to(npu_grad.dtype)

            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
            self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy())

    def test_max_pool2d_backward_case_in_ctpn(self, device):
        cpu_x = torch.rand(1, 128, 375, 500).half()
        npu_x = cpu_x.npu()
        cpu_x.requires_grad = True
        npu_x.requires_grad = True

        cpu_model = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        npu_model = copy.deepcopy(cpu_model)

        cpu_out = cpu_model(cpu_x.float()).half()
        npu_out = npu_model(npu_x)

        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))

        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())
        self.assertRtolEqual(cpu_x.grad.numpy(), npu_x.grad.cpu().numpy())


instantiate_device_type_tests(TestMaxPool2dBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()