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

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestMaxPool2dWithIndicesBackward(TestCase):

    def cpu_op_exec(self, input1, kernel_size, stride, padding, dilation, ceil_mode):
        input1.requires_grad = True
        output, _ = F.max_pool2d_with_indices(input1, kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation,
                                              ceil_mode=ceil_mode, return_indices=True)
        res = torch.sum(output)
        res.backward()
        cpu_grad = input1.grad
        output = output.detach()
        return output, cpu_grad

    def npu_op_exec(self, input1, kernel_size, stride, padding, dilation, ceil_mode):
        input1.requires_grad = True
        output, _ = F.max_pool2d_with_indices(input1, kernel_size=kernel_size, stride=stride,
                                              padding=padding, dilation=dilation,
                                              ceil_mode=ceil_mode, return_indices=True)
        res = torch.sum(output)
        res.backward()
        npu_grad = input1.grad.cpu()
        output = output.cpu().detach()
        return output, npu_grad

    def test_max_pool2d_with_indices_backward(self):
        dtype_list = [np.float32, np.float16]
        shape_list = [[256, 64, 112, 112], [1024, 24, 56, 112],
                      [1024, 24, 112, 56], [48, 48, 56], [32, 16, 43]]
        shape_format = [
            [[i, 3, j], [3, 3], [2, 2], 1, 1, False] for i in dtype_list for j in shape_list
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if item[0][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4], item[5])
            npu_output, npu_grad = self.npu_op_exec(npu_input, item[1], item[2], item[3], item[4], item[5])
            if item[0][0] == np.float16:
                cpu_output = cpu_output.to(torch.float16)
                cpu_grad = cpu_grad.to(torch.float16)

            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3)
            self.assertRtolEqual(cpu_grad, npu_grad, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
