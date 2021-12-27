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

import os
import torch
import numpy as np
from common_utils import TestCase, run_tests
import unittest
from util_test import create_common_tensor, test_2args_broadcast, create_dtype_tensor
from common_device_type import dtypes, instantiate_device_type_tests


UT_FAST_MODE = os.getenv('UT_FAST_MODE') == '1' 
class TestDiv(TestCase):
    def get_outputs(self, cpu_args, npu_args, dtype):
        # cpu not support fp16 div
        cpu_args = [i.float() if dtype==torch.half else i for i in cpu_args]
        cpu_output = torch.div(cpu_args[0], cpu_args[1]).to(dtype).numpy()
        npu_output = torch.div(npu_args[0], npu_args[1]).to("cpu").numpy()
        return cpu_output, npu_output

    def get_outputs_chk(self, cpu_args, npu_args, dtype):
        # cpu not support fp16 div
        cpu_out = torch.randn(6).to(dtype)
        npu_out = torch.randn(6).to("npu").to(dtype)
        cpu_args = [i.float() if dtype==torch.half else i for i in cpu_args]
        torch.div(cpu_args[0], cpu_args[1], out = cpu_out)
        torch.div(npu_args[0], npu_args[1], out = npu_out)
        cpu_output = cpu_out.to(dtype).numpy()
        npu_output = npu_out.to("cpu").numpy()
        return cpu_output, npu_output

    def test_div_broadcast(self, device):
        for item in test_2args_broadcast(torch.div):
            self.assertRtolEqual(item[0], item[1])

    # div not support bool
    @dtypes(torch.float, torch.half, torch.int)
    def test_div_dtype(self, device, dtype):
        cpu_input1, npu_input1 = create_dtype_tensor((2,3,4,5), dtype)
        # divisor can not be zero
        cpu_input2, npu_input2 = create_dtype_tensor((2,3,4,5), dtype, no_zero=True)
        cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], dtype)

        self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skipIf(UT_FAST_MODE, "Run UT in fast mode")
    def test_div_shape_format_fp16(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], torch.half)
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skipIf(UT_FAST_MODE, "Run UT in fast mode")
    def test_div_shape_format_fp32(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7), (2, 0, 2)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], torch.float)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_div_mix_dtype_1(self, device):
        npu_input1, npu_input2 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output, npu_output = self.get_outputs([npu_input1, npu_input3], [npu_input2, npu_input4], torch.float)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_div_mix_dtype_2(self, device):
        npu_input1, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        npu_input3 = torch.tensor(3).int()
        cpu_output, npu_output = self.get_outputs([npu_input1, npu_input3], [npu_input2, npu_input3], torch.float)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_div_mix_dtype_3(self, device):
        cpu_input1, npu_input1 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_input2 = torch.tensor(3).int()
        npu_input2 = cpu_input2.npu()
        cpu_output, npu_output = self.get_outputs([cpu_input1, cpu_input2], [npu_input1, npu_input2], torch.float)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_div_scalar_dtype(self, device):
        cpu_input1, npu_input1 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        cpu_output = cpu_input1 / 0.5
        npu_output = npu_input1 / 0.5
        self.assertRtolEqual(cpu_output, npu_output.cpu())

    @unittest.skipIf(UT_FAST_MODE, "Run UT in fast mode")
    def test_div_shape_format_fp32(self, device):
        format_list = [0, 3, 29]
        shape_list = [1, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output, npu_output = self.get_outputs_chk([cpu_input1, cpu_input2], [npu_input1, npu_input2], torch.float)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestDiv, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
