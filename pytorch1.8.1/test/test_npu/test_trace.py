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
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

LOWER = 0
UPPER = 2
INT_UPPER = 5


class TestTrace(TestCase):

    def generate_one_input(self, lower, upper, shape, dtype):
        input1 = np.random.uniform(lower, upper, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1


    def cpu_op_exec(self, input1):
        res = torch.trace(input1)
        return res.numpy()


    def cpu_op_exec_half(self, input1):
        res = torch.trace(input1)
        return res.type(torch.float16).numpy()


    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        res = torch.trace(input1)
        res = res.to("cpu")
        return res.numpy()


    def test_trace_float32(self, device):
        for shape in [(10, 10), (10, 11), (11, 10)]:
            input1 = generate_one_input(LOWER, UPPER, shape, np.float32)
            cpu_output = cpu_op_exec(input1)
            npu_output = npu_op_exec(input1)
            self.assertRtolEqual(cpu_output, npu_output)


    def test_trace_float16(self, device):
        shape = (10, 10)
        input1 = generate_one_input(LOWER, UPPER, shape, np.float16)
        cpu_output = cpu_op_exec_half(input1.type(torch.float32))
        npu_output = npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)


    def test_trace_int(self, device):
        for shape, dtype in [
            ((10, 10), np.uint8),
            ((10, 10), np.int8),
            ((10, 10), np.int32) 
        ]:
            input1 = np.random.randint(LOWER, INT_UPPER, shape, dtype)
            input1 = torch.from_numpy(input1)
            cpu_output = torch.trace(input1).numpy().astype(np.int32)
            input_npu = input1.to("npu")
            npu_output = torch.trace(input_npu)
            npu_output = npu_output.to("cpu").numpy().astype(np.int32)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestTrace, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:3")
    run_tests()
