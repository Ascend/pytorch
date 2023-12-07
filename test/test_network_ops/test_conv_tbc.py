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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConvTbc(TestCase):
    def op_exec_cpu(self, input1, weight, bias, pad):
        cpu_output = torch.conv_tbc(input1, weight, bias, pad)
        cpu_output = cpu_output.numpy().astype('float16')
        return cpu_output

    def op_exec_npu(self, input1, weight, bias, pad):
        input1 = input1.to("npu")
        weight = weight.to("npu")
        bias = bias.to("npu")
        npu_output = torch.conv_tbc(input1, weight, bias, pad)
        npu_output = npu_output.to("cpu")
        npu_output = npu_output.numpy().astype('float16')
        return npu_output

    def test_conv_tbc_shape_format(self):
        inputs = np.random.uniform(0, 2, [5, 1, 2])
        npu_input = torch.from_numpy(inputs.astype('float16'))
        cpu_input = torch.from_numpy(inputs)
        weights = np.random.uniform(0, 2, [1, 2, 2])
        npu_weight = torch.from_numpy(weights.astype('float16'))
        cpu_weight = torch.from_numpy(weights)
        bias = np.random.uniform(0, 2, [2])
        npu_bias = torch.from_numpy(bias.astype('float16'))
        cpu_bias = torch.from_numpy(bias)
        pad = 1
        cpu_output = self.op_exec_cpu(cpu_input, cpu_weight, cpu_bias, pad)
        npu_output = self.op_exec_npu(npu_input, npu_weight, npu_bias, pad)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
