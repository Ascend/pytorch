# Copyright (c) 2020, Huawei Technologies.
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


class TestRreluWithNoiseBackward(TestCase):
    def cpu_op_exec(self, input1, input2):
        input1.requires_grad_(True)
        output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
        output.backward(torch.ones_like(input1))
        res = input1.grad
        res = res.numpy()
        return res

    def npu_op_exec(self, input1, input2):
        input1.requires_grad_(True)
        output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
        output.backward(torch.ones_like(input1))
        res = input1.grad.to("cpu")
        res = res.numpy()
        return res

    def test_leaky_relu_shape_format(self):
        format_list2 = [0, 3]
        dtype_list2 = [np.float32]
        shape_list2 = [(1, 6, 4), (1, 4, 8), (1, 6, 8),
                       (2, 4, 5), (2, 5, 10), (2, 4, 10)]

        shape_format2 = [[[i, j, k]] for i in dtype_list2
                         for j in format_list2 for k in shape_list2]

        for item in shape_format2:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_leaky_relu_shape_format_fp16(self):
        format_list3 = [0, 3]
        dtype_list3 = [np.float16]
        shape_list3 = [(1, 6, 4), (1, 4, 8), (1, 6, 8),
                       (2, 4, 5), (2, 5, 10), (2, 4, 10)]

        shape_format3 = [[[i, j, k]] for i in dtype_list3
                         for j in format_list3 for k in shape_list3]

        def cpu_op_exec_fp16(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            input1.requires_grad_(True)
            output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
            output.backward(torch.ones_like(input1))
            res = input1.grad
            res = res.numpy()
            res = res.astype(np.float16)
            return res

        for item in shape_format3:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
