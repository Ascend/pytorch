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


class TestIor(TestCase):
    def generate_bool_data(self, shape):
        input1 = np.random.uniform(0, 1, shape).astype(np.float32)
        input1 = input1 < 0.5
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, input2):
        output = input1.__ior__(input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input1.__ior__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        output = input1.__ior__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test___ior___bool(self, device="npu"):
        npu_input1 = self.generate_bool_data((1, 31, 149, 2))
        npu_input2 = self.generate_bool_data((1, 31, 149, 2))
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___bool_scalar(self, device="npu"):
        npu_input1 = self.generate_bool_data((1, 31, 149, 2))
        npu_input2 = False
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_scalar(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___uint8(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 255, (1, 31, 149, 2), np.uint8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int8(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-128, 127, (1, 31, 149, 2), np.int8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_001(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, -2147483648, (1, 31, 149, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_002(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(2147483647, 2147483647, (128), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_003(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (184965, 1), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_004(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 31, 149, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_005(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (2, 31, 149, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_006(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (4, 31, 149, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_007(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (2048, 31, 1, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_008(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (8, 7, 149), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_009(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (65535, 1, 1, 1), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_010(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 8192), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_011(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 16384), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_012(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 32768), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_013(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 65535), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_014(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 131072), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_015(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 196608), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_016(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 262144), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_017(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 393216), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_018(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 524288), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_019(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 655360), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int32_020(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(-2147483648, 2147483647, (1, 1, 1, 786432), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test___ior___int_scalar(self, device="npu"):
        npu_input1 = self.generate_single_data(-2147483648, 2147483647, (1, 31, 149, 2), np.int32)
        npu_input2 = self.generate_int_scalar(-2147483648, 2147483647)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_scalar(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
