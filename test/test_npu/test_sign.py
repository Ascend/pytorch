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


class TestSign(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input= np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input)
        return npu_input

    def cpu_op_exec(self, input_x):
        output = torch.sign(input_x)
        output = output.numpy()
        return output

    def npu_op_exec(self, input_x):
        input = input_x.to("npu")
        output= torch.sign(input)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_sign_float16(self, device):
        def cpu_op_exec_fp16(input):
            input = input.to(torch.float32)
            output = torch.sign(input)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        npu_input = self.generate_data(0, 100, (5,3), np.float16)
        cpu_output = cpu_op_exec_fp16(npu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_sign_float32(self, device):
        npu_input = self.generate_data(0, 100, (4,3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input)
        npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestSign, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
