# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMiopenConvolutionTranspose(TestCase):
    def cpu_op_exec(self, input, weight):
        cpu_output = torch.nn.functional.conv_transpose2d(input, weight, padding = 1)
        print("===cpu_output===")
        print(cpu_output)

        return cpu_output

    def npu_op_exec(self, input, weight):
        input = input.to("npu")
        weight = weight.to("npu")
        npu_output = torch.nn.functional.conv_transpose2d(input, weight, padding = 1)
        npu_output = npu_output.to("cpu")
        print("===npu_output===")
        print(npu_output)

        return npu_output


    def test_conv_transpose2d_float32(self, device):
        inputs = torch.randn(1, 4, 5, 5)
        weights = torch.randn(4, 4, 3, 3)
        cpu_output = self.cpu_op_exec(inputs, weights)
        npu_output = self.npu_op_exec(inputs, weights)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestMiopenConvolutionTranspose, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
