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

import time
import unittest

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.function import roll

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestRoll(TestCase):
    def npu_slow_roll_op_exec(self, input1, shift_size, dims):
        output = torch.roll(input1, shifts=(-shift_size, -shift_size), dims=dims)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            torch.roll(input1, shifts=(-shift_size, -shift_size), dims=dims)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output.to("cpu").numpy(), slow_time

    def npu_fast_roll_op_exec(self, input1, shift_size, dims):
        output = roll(input1, shifts=(-shift_size, -shift_size), dims=dims)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            roll(input1, shifts=(-shift_size, -shift_size), dims=dims)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output.to("cpu").numpy(), fast_time

    def test_roll_shape_format(self):
        dtype_list = [np.float16, np.float32, np.uint8, np.int32]
        format_list = [-1, 2]
        shape_list = [[32, 56, 56, 16]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            _, npu_input = create_common_tensor(item, -10, 10)
            shift_size = 3
            slow_output, slow_time = self.npu_slow_roll_op_exec(npu_input, shift_size, (1, 2))
            fast_output, fast_time = self.npu_fast_roll_op_exec(npu_input, shift_size, (1, 2))
            self.assertRtolEqual(slow_output, fast_output)
            if item[0] == np.uint8:
                continue
            self.assertTrue(slow_time > fast_time)

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "it's OK on Ascend910B only.")
    def test_roll_shape_format_uint8(self):
        dtype_list = [np.uint8]
        format_list = [-1, 2]
        shape_list = [[32, 56, 56, 16]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            _, npu_input = create_common_tensor(item, -10, 10)
            shift_size = 3
            slow_output, slow_time = self.npu_slow_roll_op_exec(npu_input, shift_size, (1, 2))
            fast_output, fast_time = self.npu_fast_roll_op_exec(npu_input, shift_size, (1, 2))
            self.assertTrue(slow_time > fast_time)


if __name__ == "__main__":
    run_tests()
