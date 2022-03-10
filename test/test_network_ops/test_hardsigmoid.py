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
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests

class TestHardsigmoid(TestCase):
    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, input2):
        output = input2
        h = torch.nn.Hardsigmoid()
        output = h(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        output = input2
        h = torch.nn.Hardsigmoid()
        output = h(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_hardsigmoid_int32(self, device="npu"):
        def cpu_op_exec_int32(input1):
            input1 = input1.to(torch.float32)
            h = torch.nn.Hardsigmoid()
            output = h(input1)
            output = output.numpy()
            output = output.astype(np.int32)
            return output
        npu_input1 = self.generate_single_data(-6, 6, (3,6), np.int32)
        npu_input2 = self.generate_single_data(-6, 6, (3,6), np.int32)
        cpu_output = cpu_op_exec_int32(npu_input1)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_hardsigmoid_float32(self, device="npu"):
        npu_input1 = self.generate_single_data(-6, 6, (9,3), np.float32)
        npu_input2 = self.generate_single_data(-6, 6, (9,3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_hardsigmoid_float16(self, device="npu"):
        def cpu_op_exec_float16(input1):
            input1 = input1.to(torch.float32)
            h = torch.nn.Hardsigmoid()
            output = h(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
        npu_input1 = self.generate_single_data(-6, 6, (2,7), np.float16)
        npu_input2 = self.generate_single_data(-6, 6, (2,7), np.float16)
        cpu_output = cpu_op_exec_float16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

if __name__ == '__main__':
    run_tests()