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
import math
import random
from torch._six import nan
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests


class TestNormExceptDim(TestCase):
    def generate_data(self, min, max, shape, dtype):
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        input2 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_single_data(self, min, max, shape, dtype):
        input = np.random.uniform(min, max, shape).astype(dtype)
        npu_input = torch.from_numpy(input)
        return npu_input

    def generate_int_dim(self, max):
        dim = np.random.randint(0, max)
        return dim

    def generate_bool_keepdim(self):
        keepdim = random.choice([True, False])
        return keepdim

    def test_norm_except_dim_type(self, device):
        def cpu_op_exec(input1, pow):
            output = torch.norm_except_dim(input1, pow=pow, dim=0)
            output = output.numpy()
            return output

        def npu_op_exec(input1, pow):
            print(input1.shape)
            input1 = input1.to("npu")
            output = torch.norm_except_dim(input1, pow=pow, dim=0)
            output = output.to("cpu")
            output = output.numpy()
            print(output.shape)
            return output

        def test_norm_except_dim_exec(input_type):
            input1 = self.generate_single_data(0, 100, (5, 3), input_type)
            pow = self.generate_int_dim(10)
            cpu_output = cpu_op_exec(input1, pow)
            npu_output = npu_op_exec(input1, pow)
            return cpu_output, npu_output

        for dtype in [np.float32]:
            cpu_output, npu_output = test_norm_except_dim_exec(dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    
instantiate_device_type_tests(TestNormExceptDim, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()