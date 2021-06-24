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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestBartlettWindow(TestCase):

    def cpu_op_exec_length(self, length):
        output = torch.bartlett_window(length, dtype=torch.float32)
        output = output.numpy()
        return output

    def cpu_op_exec_periodic(self, length, periodic):
        output = torch.bartlett_window(length, periodic, dtype=torch.float32)
        output = output.numpy()
        return output
    
    def npu_op_exec_length(self, length):
        d = torch.device("npu")
        output = torch.bartlett_window(length, device=d)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def npu_op_exec_periodic(self, length, periodic):
        d = torch.device("npu")
        output = torch.bartlett_window(length, periodic, device=d)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def subtest_bartlett_window_length(self, length):
        cpu_output = self.cpu_op_exec_length(length)
        npu_output = self.npu_op_exec_length(length)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def subtest_bartlett_window_periodic(self, length, periodic):
        cpu_output = self.cpu_op_exec_periodic(length, periodic)
        npu_output = self.npu_op_exec_periodic(length, periodic)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_bartlett_window(self, device):
        self.subtest_bartlett_window_length(0)
        self.subtest_bartlett_window_length(78)
        self.subtest_bartlett_window_length(6)
        self.subtest_bartlett_window_length(1)
        self.subtest_bartlett_window_length(345632)
        self.subtest_bartlett_window_length(4214748)
        self.subtest_bartlett_window_length(6784)
        self.subtest_bartlett_window_length(214748)
        self.subtest_bartlett_window_periodic(214748, True)
        self.subtest_bartlett_window_periodic(214748, False)
        self.subtest_bartlett_window_periodic(6, True)
        self.subtest_bartlett_window_periodic(6, False)
        self.subtest_bartlett_window_periodic(1, True)
        self.subtest_bartlett_window_periodic(1, False)
        self.subtest_bartlett_window_periodic(0, False)
        self.subtest_bartlett_window_periodic(0, True)


instantiate_device_type_tests(TestBartlettWindow, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:0")
    run_tests()
