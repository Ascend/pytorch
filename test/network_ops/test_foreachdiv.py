# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices


class TestAdd(TestCase):

    def cpu_op_exec(self, input1, scalar):
        output = torch._foreach_div(input1, scalar)
        return output

    def npu_op_exec(self, input1, scalar):
        output = torch._foreach_div(input1, scalar)
        return output

    def cpu_op_exec_alpha(self, input1, input2):
        output = torch._foreach_div(input1, input2)
        return output

    def npu_op_exec_alpha(self, input1, input2):
        output = torch._foreach_div(input1, input2)
        return output

    def cpu_op_scalar_exec(self, input1, scalararray):
        output = torch._foreach_div(input1, scalararray)
        return output

    def npu_op_scalar_exec(self, input1, scalararray):
        output = torch._foreach_div(input1, scalararray)
        return output

    def cpu_op_exec_(self, input1, scalar):
        torch._foreach_div_(input1, scalar)
        return input1

    def npu_op_exec_(self, input1, scalar):
        torch._foreach_div_(input1, scalar)
        return input1

    def cpu_op_exec_alpha_(self, input1, input2):
        torch._foreach_div_(input1, input2)
        return input1

    def npu_op_exec_alpha_(self, input1, input2):
        torch._foreach_div_(input1, input2)
        return input1

    def cpu_op_scalar_exec_(self, input1, scalararray):
        torch._foreach_div_(input1, scalararray)
        return input1

    def npu_op_scalar_exec_(self, input1, scalararray):
        torch._foreach_div_(input1, scalararray)
        return input1

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_1d(self):
        shape_format = [
            [[np.float16, 2, [50]], [np.float16, 2, [50]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_1d(self):
        shape_format = [
            [[np.float32, 2, [50]], [np.float32, 2, [50]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_2d(self):
        shape_format = [
            [[np.float16, 2, [50, 25]], [np.float16, 2, [50, 25]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_2d(self):
        shape_format = [
            [[np.float32, 2, [50, 25]], [np.float32, 2, [50, 25]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_3d(self):
        shape_format = [
            [[np.float16, 2, [50, 25, 7]], [np.float16, 2, [50, 25, 7]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_3d(self):
        shape_format = [
            [[np.float32, 2, [50, 25, 7]], [np.float32, 2, [50, 25, 7]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_4d(self):
        shape_format = [
            [[np.float16, 2, [50, 25, 7, 100]], [np.float16, 2, [50, 25, 7, 100]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_4d(self):
        shape_format = [
            [[np.float32, 2, [50, 25, 7, 100]], [np.float32, 2, [50, 25, 7, 100]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_1d_(self):
        shape_format = [
            [[np.float16, 2, [50]], [np.float16, 2, [50]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_1d_(self):
        shape_format = [
            [[np.float32, 2, [50]], [np.float32, 2, [50]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_2d_(self):
        shape_format = [
            [[np.float16, 2, [50, 25]], [np.float16, 2, [50, 25]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_2d_(self):
        shape_format = [
            [[np.float32, 2, [50, 25]], [np.float32, 2, [50, 25]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_3d_(self):
        shape_format = [
            [[np.float16, 2, [50, 25, 7]], [np.float16, 2, [50, 25, 7]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_3d_(self):
        shape_format = [
            [[np.float32, 2, [50, 25, 7]], [np.float32, 2, [50, 25, 7]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp16_4d_(self):
        shape_format = [
            [[np.float16, 2, [50, 25, 7, 100]], [np.float16, 2, [50, 25, 7, 100]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())

    @SupportedDevices(['Ascend910B'])
    def test_add_scalar_shape_format_fp32_4d_(self):
        shape_format = [
            [[np.float32, 2, [50, 25, 7, 100]], [np.float32, 2, [50, 25, 7, 100]]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output1 = self.cpu_op_exec_([cpu_input1, cpu_input2], 1.0)
            npu_output1 = self.npu_op_exec_([npu_input1, npu_input2], 1.0)
            cpu_output2 = self.cpu_op_exec_alpha_([cpu_input1, cpu_input2], [cpu_input2, cpu_input1])
            npu_output2 = self.npu_op_exec_alpha_([npu_input1, npu_input2], [npu_input2, npu_input1])
            cpu_output3 = self.cpu_op_scalar_exec_([cpu_input1, cpu_input2], [1.0, 2.0])
            npu_output3 = self.npu_op_scalar_exec_([npu_input1, npu_input2], [1.0, 2.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

            for (cpu_tmp3, npu_tmp3) in zip(cpu_output3, npu_output3):
                self.assertRtolEqual(cpu_tmp3.numpy(), npu_tmp3.to("cpu").numpy())


if __name__ == "__main__":
    run_tests()
