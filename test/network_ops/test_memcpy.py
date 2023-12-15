# Copyright (c) 2022, Huawei Technologies.
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


class TestMemcpy(TestCase):

    def test_copy_memory_(self):
        def cpu_op_exec(input1, input2):
            out_mul = torch.mul(input1, input2)
            out_add = torch.add(input1, input2)
            out_mul.copy_(out_add)
            out = torch.sub(out_mul, input2)
            return out.numpy()

        def npu_op_exec(input1, input2):
            out_mul = torch.mul(input1, input2)
            out_add = torch.add(input1, input2)
            out_mul.copy_memory_(out_add)
            out = torch.sub(out_mul, input2)
            return out.to("cpu").numpy()

        dtype_list = [np.int32, np.float32]
        format_list = [0, 3, 29]
        shape_list = [
            [9, 13],
            [3, 16, 5, 5],
        ]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in dtype_shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output = cpu_op_exec(cpu_input, 2)
            npu_output = npu_op_exec(npu_input, 2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_h2d_inplace(self):
        def cpu_copy_op_exec(input1_host, input2_host):
            input1_host.add_(input1_host)
            input1_host.copy_(input2_host)
            return input1_host.numpy()

        def npu_copy_op_exec(input1_device, input2_host):
            input1_device.add_(input1_device)
            input1_device.copy_(input2_host)
            return input1_device.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (5, 5)], 1, 100)
        cpu_input2 = cpu_input1 + 1
        cpu_output = cpu_copy_op_exec(cpu_input1, cpu_input2)
        npu_output = npu_copy_op_exec(npu_input1, cpu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_d2h_inplace(self):
        def cpu_copy_op_exec(input1_host, input2_host):
            input1_host.add_(input1_host)
            input2_host.copy_(input1_host)
            return input2_host.numpy()

        def npu_copy_op_exec(input1_device, input2_host):
            input1_device.add_(input1_device)
            input2_host.copy_(input1_device)
            return input2_host.numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (5, 5)], 1, 100)
        cpu_out1 = cpu_input1 + 1
        cpu_out2 = cpu_input1 + 1
        cpu_output = cpu_copy_op_exec(cpu_input1, cpu_out1)
        npu_output = npu_copy_op_exec(npu_input1, cpu_out2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_item(self):
        def cpu_copy_op_exec(input1, input2):
            input1.add_(input2)
            out = input1 * input2
            out = out.item()
            return out

        def npu_copy_op_exec(input1, input2):
            input1.add_(input2)
            out = input1 * input2
            out = out.item()
            return out

        dtype_list = [np.int32, np.float32]
        format_list = [-1]
        shape_list = [
            [1],
            [1, 1, 1, ],
        ]
        dtype_shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]

        for item in dtype_shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output = cpu_copy_op_exec(cpu_input, 2) + cpu_copy_op_exec(cpu_input, 3)
            npu_output = npu_copy_op_exec(npu_input, 2) + npu_copy_op_exec(npu_input, 3)
            self.assertEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
