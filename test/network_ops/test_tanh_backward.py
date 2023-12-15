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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTanhBackward(TestCase):

    def cpu_op_exec(self, input1):
        input1.requires_grad = True
        input1_tanh = torch.tanh(input1)
        input1_tanh.backward(torch.ones_like(input1_tanh))
        output = input1.grad.numpy()
        return output

    def npu_op_exec(self, input1):
        input1.requires_grad = True
        input1_tanh = torch.tanh(input1)
        input1_tanh.backward(torch.ones_like(input1_tanh))
        output = input1.grad
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_tanh_backward_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (4, 3)], 1, 100],
            [[np.float32, -1, (7, 5, 5)], 21474836, 21474837],
            [[np.float32, -1, (4, 44, 44)], 3450, 34020],
            [[np.float32, -1, (65500, 3, 3)], -214748, -214746],
            [[np.float32, -1, (1024, 448, 448)], 200, 300],
            [[np.float32, -1, (24, 24, 3)], -2, -2],
            [[np.float32, -1, (3, 7, 7)], 0.3793216987112159, 1],
            [[np.float32, -1, (2, 8, 8)], 0.9662927186969077, 1],
            [[np.float32, -1, (3, 7, 7)], 0.9956475043306917, 2],
            [[np.float32, -1, (7, 10, 10)], 0.769565434387681, 3],
            [[np.float32, -1, (65500, 1, 1)], 95, 100],
            [[np.float32, -1, (6, 3, 10)], 0.03133650248813469, 2],
            [[np.float32, -1, (4, 3, 3, 3, 3, 3, 3, 3)], 0, 1],
            [[np.float32, -1, (5,)], 0, 1],
            [[np.float32, -1, (5, 5, 5, 5, 5, 5)], 1, 2],
            [[np.float32, -1, (5, 5, 5, 5, 5, 5)], 2, 3],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tanh_backward_float16_shape_format(self, device="npu"):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            input1.requires_grad = True
            input1_tanh = torch.tanh(input1)
            input1_tanh.backward(torch.ones_like(input1_tanh))
            output = input1.grad.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (65500, 1)], 212, 225],
            [[np.float16, -1, (1024, 448, 448)], 200, 300],
            [[np.float16, -1, (16, 16)], -1000, -100],
            [[np.float16, -1, (4, 1)], -1.1754943508e-38, -1.1754943508e-38],
            [[np.float16, -1, (7, 5, 5)], 21474836, 21474837],
            [[np.float16, -1, (4, 44, 44)], 3450, 34020],
            [[np.float16, -1, (65500, 3, 3)], -214748, -214746],
            [[np.float16, -1, (64, 4, 4)], -9.313225746154785e-10, 9.313225746154785e-10],
            [[np.float16, -1, (128, 3, 5)],
             -0.000000000000000000000000000000000000011754943508,
             0.000000000000000000000000000000000000011754943508],
            [[np.float16, -1, (65500, 1, 1)], 95, 100],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
