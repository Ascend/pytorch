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


class TestRreluWithNoise(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch._C._nn.rrelu_with_noise(input1, input2, 0.1, 0.3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        torch._C._nn.rrelu_with_noise_(input1, input2, 0.1, 0.3)
        output = input1.cpu()
        output = output.numpy()
        return output

    def test_leaky_relu_shape_format(self):
        format_list = [0, 3]
        dtype_list = [np.float32, np.float16]
        shape_list = [(1, 6, 4), (1, 4, 8), (1, 6, 8),
                      (2, 4, 5), (2, 5, 10), (2, 4, 10)]

        shape_format = [[[i, j, k]] for i in dtype_list
                        for j in format_list for k in shape_list]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.float()
                cpu_input2 = cpu_input2.float()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_leaky_relu_inplace_shape_format(self):
        format_list = [0, 3]
        dtype_list = [np.float32, np.float16]
        shape_list = [(1, 6, 4), (2, 5, 10), (2, 4, 10)]
        shape_format = [[[i, j, k]] for i in dtype_list
                        for j in format_list for k in shape_list]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.float()
                cpu_input2 = cpu_input2.float()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
