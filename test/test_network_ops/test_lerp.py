# Copyright (c) 2020 Huawei Technologies Co., Ltd
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


import itertools
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLerp(TestCase):

    def cpu_op_exec(self, input1, input2, input3):
        output = torch.lerp(input1, input2, input3)
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1, input2, input3):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        input3 = input3.to(torch.float32)
        output = torch.lerp(input1, input2, input3)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_exec(self, input1, input2, input3):
        output = torch.lerp(input1, input2, input3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1)
        torch.lerp(input1, input2, input3, out=output)
        output = output.numpy()
        return output

    def cpu_op_out_exec_fp16(self, input1, input2, input3):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        input3 = input3.to(torch.float32)
        output = torch.ones_like(input1)
        torch.lerp(input1, input2, input3, out=output)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1)
        torch.lerp(input1, input2, input3, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_scalar_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1)
        torch.lerp(input1, input2, input3, out=output)
        output = output.numpy()
        return output

    def cpu_op_scalar_exec_fp16(self, input1, input2, input3):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = torch.lerp(input1, input2, input3)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def cpu_op_scalar_out_exec_fp16(self, input1, input2, input3):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = torch.ones_like(input1)
        torch.lerp(input1, input2, input3, out=output)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_scalar_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1)
        torch.lerp(input1, input2, input3, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_lerp_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (4, 2, 2, 3)]],
            [[np.float32, -1, (2, 2, 3, 4)]],
            [[np.float32, -1, (3, 3, 3)]],
            [[np.float32, -1, (4, 4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            cpu_output1 = self.cpu_op_out_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output1 = self.npu_op_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_lerp_float16_shape_format(self):
        shape_format = [
            [[np.float16, -1, (100, 4, 5, 5)]],
            [[np.float16, -1, (100, 5, 5, 4)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 10, 100)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            cpu_output1 = self.cpu_op_out_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output1 = self.npu_op_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output, prec=0.003, prec16=0.003)
            self.assertRtolEqual(cpu_output1, npu_output1, prec=0.003, prec16=0.003)

    def test_lerp_scalar_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (4, 2, 2, 3)], 1.0],
            [[np.float32, -1, (2, 2, 3, 4)], 2.0],
            [[np.float32, -1, (3, 3, 3)], 1.2],
            [[np.float32, -1, (4, 4, 4)], 1.2]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_input3 = item[1]
            npu_input3 = item[1]
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            cpu_output1 = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output1 = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_lerp_scalar_float16_shape_format(self):
        shape_format = [
            [[np.float16, -1, (100, 4, 5, 5)], 1.2],
            [[np.float16, -1, (100, 5, 5, 4)], 1.2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_input3 = item[1]
            npu_input3 = item[1]
            cpu_output = self.cpu_op_scalar_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            cpu_output1 = self.cpu_op_scalar_out_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output1 = self.npu_op_scalar_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output, prec16=0.02)
            self.assertRtolEqual(cpu_output1, npu_output1, prec16=0.02)

    def test_lerp_broadcast_shape_format(self):
        shape_list = [
            [],
            [5, ],
            [5, 5],
        ]
        for shapes in itertools.product(shape_list, shape_list):
            cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, shapes[0]], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor([np.float32, -1, shapes[1]], 10, 100)
            cpu_input3, npu_input3 = create_common_tensor([np.float32, -1, shapes[0]], 10, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            cpu_output1 = self.cpu_op_out_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output1 = self.npu_op_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_lerp_inplace_shape_format(self):
        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, [2, 1]], 10, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, -1, [2, 6]], 10, 100)
        cpu_input2.lerp_(cpu_input1, 1)
        npu_input2.lerp_(npu_input1, 1)
        self.assertRtolEqual(cpu_input2, npu_input2.cpu())

        def lerp_inplace(npu_input1, npu_input2):
            npu_input1.lerp_(npu_input2, 1)
        self.assertRaisesRegex(
            Exception, "doesn't match the broadcast shape", lerp_inplace, npu_input1, npu_input2)


if __name__ == '__main__':
    run_tests()
