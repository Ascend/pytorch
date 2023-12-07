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


class TestBatchNorm(TestCase):
    def cpu_op_exec(self, input1, num_features, affine):
        flag = False
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            flag = True
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
        output = m(input1)
        if flag:
            output = output.to(torch.float16)
        output_cpu = output.detach().numpy()
        return output_cpu

    def npu_op_exec_new(self, input1, num_features, affine):
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
        m = m.to("npu")
        output = m(input1)
        output = output.to("cpu").detach().numpy()
        return output

    def test_batchnorm_shape_format(self):
        format_list = [-1, 3, 0]
        shape_list = [(10, 32, 35, 45), (256, 100, 7, 7), (256, 100, 14, 14), (10, 56, 28, 28), (10, 56, 56, 56)]
        affine_list = [True]
        dtype_list = [np.float16, np.float32]
        shape_format = [
            [[z, i, j], h] for z in dtype_list for i in format_list for j in shape_list for h in affine_list
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_output = self.cpu_op_exec(cpu_input1, item[0][2][1], item[1])
            npu_output = self.npu_op_exec_new(npu_input1, item[0][2][1], item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec_3d(self, input1, num_features, affine):
        flag = False
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            flag = True
        m = torch.nn.BatchNorm3d(num_features, affine=affine)
        output = m(input1)
        if flag:
            output = output.to(torch.float16)
        output_cpu = output.detach().numpy()
        return output_cpu

    def npu_op_exec_new_3d(self, input1, num_features, affine):
        m = torch.nn.BatchNorm3d(num_features, affine=affine)
        m = m.to("npu")
        output = m(input1)
        output = output.to("cpu").detach().numpy()
        return output

    def test_batchnorm_shape_format_3d(self):
        format_list = [-1]
        shape_list = [[8, 512, 4, 28, 28], [8, 256, 8, 56, 56]]
        affine_list = [True]
        dtype_list = [np.float16, np.float32]
        shape_format = [
            [[z, i, j], h] for z in dtype_list for i in format_list for j in shape_list for h in affine_list
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_output = self.cpu_op_exec_3d(cpu_input1, item[0][2][1], item[1])
            npu_output = self.npu_op_exec_new_3d(npu_input1, item[0][2][1], item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
