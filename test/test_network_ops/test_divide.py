# Copyright (c) 2022, Huawei Technologies.All rights reserved.
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

import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, test_2args_broadcast, create_dtype_tensor
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestDivide(TestCase):
    def get_outputs(self, cpu_args, npu_args, dtype):
        # cpu not support fp16 div
        cpu_args = [i.float() if dtype == torch.half else i for i in cpu_args]
        cpu_output = torch.divide(cpu_args[0], cpu_args[1]).to(dtype).numpy()
        npu_output = torch.divide(npu_args[0], npu_args[1]).to("cpu").numpy()
        return cpu_output, npu_output

    def get_outputs_chk(self, cpu_args, npu_args, dtype):
        # cpu not support fp16 div
        cpu_out = torch.randn(6).to(dtype)
        npu_out = torch.randn(6).to("npu").to(dtype)
        cpu_args = [i.float() if dtype == torch.half else i for i in cpu_args]
        torch.divide(cpu_args[0], cpu_args[1], out=cpu_out)
        torch.divide(npu_args[0], npu_args[1], out=npu_out)
        cpu_output = cpu_out.to(dtype).numpy()
        npu_output = npu_out.to("cpu").numpy()
        return cpu_output, npu_output

    def test_divide_broadcast(self):
        for item in test_2args_broadcast(torch.divide):
            self.assertRtolEqual(item[0], item[1])

    # divide not support bool
    @Dtypes(torch.float, torch.half, torch.int)
    def test_divide_dtype(self, dtype):
        cpu_input1, npu_input1 = create_dtype_tensor((2, 3, 4, 5), dtype)
        cpu_input2, npu_input2 = create_dtype_tensor((2, 3, 4, 5), dtype, no_zero=True)
        cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], dtype)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_divide_shape_format_fp16(self):
        format_list = [0, 3, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format1 = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], torch.half)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_divide_shape_format_fp32(self):
        format_list = [0, 3, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7), (2, 0, 2)]
        shape_format1 = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], torch.float)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_divide_mix_dtype_1(self):
        npu_input1, npu_input2 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output, npu_output = self.get_outputs([npu_input1, npu_input3], [npu_input2, npu_input4], torch.float)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_divide_mix_dtype_2(self):
        npu_input1, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        npu_input3 = torch.tensor(3).int()
        cpu_output, npu_output = self.get_outputs([npu_input1, npu_input3], [npu_input2, npu_input3], torch.float)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_divide_scalar_dtype(self):
        cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        cpu_output = cpu_input1 / 0.5
        npu_output = npu_input1 / 0.5
        self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_divide_npuscalar_dtype(self):
        cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        cpu_output1 = cpu_input1 / torch.tensor(0.5)
        npu_output1 = npu_input1 / torch.tensor(0.5).npu()
        self.assertRtolEqual(cpu_output1, npu_output1.cpu())

    def test_divide_shape_format_fp32_1(self):
        format_list = [0, 3, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format1 = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output, npu_output = self.get_outputs_chk([cpu_input1, cpu_input2],
                                                          [npu_input1, npu_input2], torch.float)
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec_mode(self, input1, input2, mode):
        output = torch.divide(input1, input2, rounding_mode=mode)
        return output

    def npu_op_exec_mode(self, input1, input2, mode):
        output = torch.divide(input1, input2, rounding_mode=mode)
        return output.cpu()

    def npu_op_exec_mode_out(self, input1, input2, output, mode):
        torch.divide(input1, input2, rounding_mode=mode, out=output)
        return output.cpu()

    def npu_op_exec_mode_inp(self, input1, input2, mode):
        input1.divide_(input2, rounding_mode=mode)
        return input1.cpu()

    def test_divide_tensor_mode(self):
        shape_format1 = [
            [[np.float32, 0, (20, 16)], [np.float32, 0, (16)], [np.float32, 0, (20, 16)], 'floor'],
            [[np.float32, 0, (20, 16)], [np.float32, 0, (20, 16)], [np.float32, 0, (20, 16)], 'trunc'],
            [[np.float16, 0, (2, 20, 16)], [np.float16, 0, (16)], [np.float16, 0, (2, 20, 16)], 'trunc'],
            [[np.float16, 0, (2, 20, 16)], [np.float16, 0, (20, 16)], [np.float16, 0, (2, 20, 16)], 'floor'],
            [[np.float16, 0, (3, 20, 16)], [np.float16, 0, (20, 16)], [np.float16, 0, (3, 20, 16)], 'true'],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            # divide
            cpu_output = self.cpu_op_exec_mode(cpu_input1, cpu_input2, item[3])
            npu_output = self.npu_op_exec_mode(npu_input1, npu_input2, item[3])
            self.assertRtolEqual(cpu_output, npu_output)
            # divide_out
            cpu_out, npu_out = create_common_tensor(item[2], 1, 100)
            npu_output_out = self.npu_op_exec_mode_out(npu_input1, npu_input2, npu_out, item[3])
            self.assertRtolEqual(cpu_output, npu_output_out)
            # divide_
            npu_output_inp = self.npu_op_exec_mode_inp(npu_input1, npu_input2, item[3])
            self.assertRtolEqual(cpu_output, npu_output_inp)

    def test_divide_scalar_mode(self):
        shape_format = [
            [[np.float32, 0, (20, 16)], 15.9, 'floor'],
            [[np.float32, 0, (20, 16)], 17.2, 'trunc'],
            [[np.float16, 0, (2, 20, 16)], 72.2, 'floor'],
            [[np.float16, 0, (2, 20, 16)], -5.4, 'trunc'],
            [[np.float16, 0, (3, 20, 16)], -45.3, 'true'],
            [[np.int32, 0, (20, 16)], 15.9, 'floor'],
            [[np.int32, 0, (20, 16)], 17.2, 'trunc'],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            # divide
            cpu_output = self.cpu_op_exec_mode(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec_mode(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)
            # divide_
            npu_output_inp = self.npu_op_exec_mode_inp(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output_inp)


if __name__ == "__main__":
    run_tests()
