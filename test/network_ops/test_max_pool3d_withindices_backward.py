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
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaxPool3dWithIndicesBackward(TestCase):
    def cpu_op_exec(self, inputCpu, kernel_size, stride, padding, dilation, ceil_mode):
        inputCpu.requires_grad = True
        m = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                               ceil_mode=ceil_mode)
        dataCpu = m(inputCpu)
        z1 = torch.sum(dataCpu)
        ones = torch.ones_like(z1)
        z1.backward(ones)
        cpu_grad = inputCpu.grad
        output1 = dataCpu.detach()
        return output1, cpu_grad

    def npu_op_exec(self, inputNpu, kernel_size, stride, padding, dilation, ceil_mode):
        inputNpu.requires_grad = True
        m = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                               ceil_mode=ceil_mode)
        dataNpu = m(inputNpu)
        z2 = torch.sum(dataNpu)
        ones = torch.ones_like(z2)
        z2.backward(ones)
        npu_grad = inputNpu.grad
        npu_grad = npu_grad.to("cpu")
        output1 = dataNpu.to("cpu").detach()
        return output1, npu_grad

    def test_max_pool3d_backward_shape_format(self):
        shape_format = [
            [np.float16, 30, [1, 3, 19, 19, 19], [3, 3, 3], [2, 2, 2], 1, 1, False],
            [np.float16, 30, [3, 3, 124, 112, 112], 3, [2, 2, 2], 1, 1, True],
            [np.float16, 30, [10, 64, 56, 56, 56], 5, 2, [2, 2, 2], 1, True],
            [np.float16, 30, [10, 10, 10, 10, 10], 3, 2, 1, 1, False],
            [np.float16, 30, [64, 10, 124, 56, 64], 3, 2, 0, 1, False],
            [np.float16, 30, [64, 10, 64, 32, 32], 3, 2, 0, 1, True]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:3], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, item[3], item[4], item[5], item[6], item[7])
            npu_output, npu_grad = self.npu_op_exec(npu_input, item[3], item[4], item[5], item[6], item[7])
            cpu_output = cpu_output.to(npu_output.dtype)
            cpu_grad = cpu_grad.to(npu_grad.dtype)

            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
            self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy())


if __name__ == "__main__":
    run_tests()
