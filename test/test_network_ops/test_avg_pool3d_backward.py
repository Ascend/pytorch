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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAvgPool3DBackward(TestCase):

    def cpu_op_exec(self, kernel_size, stride, input1):
        m = torch.nn.AvgPool3d(kernel_size, stride)
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output_grad, output

    def cpu_op_exec_fp16(self, kernel_size, stride, input1):
        m = torch.nn.AvgPool3d(kernel_size, stride)
        input1.requires_grad = True
        output = m(input1.float())
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.detach().numpy()
        output = output.half()
        output = output.detach().numpy()
        return output_grad, output

    def npu_op_exec(self, kernel_size, stride, input1):
        m = torch.nn.AvgPool3d(kernel_size, stride).npu()
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.to("cpu")
        output = output.detach().numpy()
        return output_grad, output

    def test_avg_pool_3d_fp32(self):
        shape_format = [
            [[np.float32, -1, (20, 16, 50, 44, 31)], (3, 2, 2), (2, 1, 2)],
            [[np.float32, -1, (2, 1, 4, 4, 4)], 3, 2],
            [[np.float32, -1, (2, 1, 4, 4, 4)], 2, 2],
            [[np.float32, -1, (2, 4, 4, 4)], 2, 2]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_output_grad, npu_output = self.npu_op_exec(item[1], item[2], npu_input1)
            cpu_output_grad, cpu_output = self.cpu_op_exec(item[1], item[2], cpu_input1)
            self.assertRtolEqual(cpu_output, npu_output, 1.e-3)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad, 1.e-3)

    def test_avg_pool_3d_fp16(self):
        shape_format = [
            [[np.float16, -1, (20, 16, 50, 44, 31)], (3, 2, 2), (2, 1, 2)],
            [[np.float16, -1, (2, 1, 4, 4, 4)], 3, 2],
            [[np.float16, -1, (2, 1, 4, 4, 4)], 2, 2],
            [[np.float16, -1, (2, 4, 4, 4)], 2, 2]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            npu_output_grad, npu_output = self.npu_op_exec(item[1], item[2], npu_input1)
            cpu_output_grad, cpu_output = self.cpu_op_exec_fp16(item[1], item[2], cpu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)


if __name__ == "__main__":
    run_tests()
