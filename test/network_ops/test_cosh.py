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


class TestCosh(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1):
        output = torch.cosh(input1)
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1):
        input1 = input1.to(torch.float32)
        output = torch.cosh(input1)
        output = output.numpy().astype(np.float16)
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.cosh(input1)
        output = output.to("cpu").numpy()
        return output

    def npu_out_op_exec(self, input1, output1):
        input1 = input1.to("npu")
        output1 = output1.to("npu")
        torch.cosh(input1, out=output1)
        output = output1.to("cpu").numpy()
        return output

    def test_cosh_float16_1(self):
        npu_input1 = self.generate_single_data(-2, 2,
                                               ((65535, 1, 1, 1)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float16_2(self):
        npu_input1 = self.generate_single_data(-2, 2,
                                               ((1, 1, 1, 8192)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float16_3(self):
        npu_input1 = self.generate_single_data(-2, 2,
                                               ((1, 1, 1, 65535)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float16_4(self):
        npu_input1 = self.generate_single_data(-2, 2,
                                               ((1, 1, 1, 524288)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float16_5(self):
        npu_input1 = self.generate_single_data(-2, 2,
                                               ((1, 1, 1, 786432)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float16_6(self):
        npu_input1 = self.generate_single_data(-5, 5,
                                               ((1, 1, 1, 786432)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_1(self):
        npu_input1 = self.generate_single_data(-1.1754943508e-38,
                                               -1.1754943508e-38, ((1, 31, 149, 2)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_2(self):
        npu_input1 = self.generate_single_data(-0.000030517578125,
                                               0.000030517578125, ((2, 32, 149, 31)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_3(self):
        npu_input1 = self.generate_single_data(-9.313225746154785e-10,
                                               9.313225746154785e-10, ((184965, 1)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_4(self):
        npu_input1 = self.generate_single_data(-3, 3, ((1, 31, 149, 2)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_5(self):
        npu_input1 = self.generate_single_data(-9.313225746154785e-10,
                                               9.313225746154785e-10, ((1, 31, 149, 2)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_6(self):
        npu_input1 = self.generate_single_data(-0.000000000000000000000000000000000000011754943508,
                                               0.000000000000000000000000000000000000011754943508,
                                               ((2, 31, 149, 2)),
                                               np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_7(self):
        npu_input1 = self.generate_single_data(0.000000000000000000000000000000000000011754943508,
                                               0.000000000000000000000000000000000000011754943508,
                                               ((4, 31, 149, 2)),
                                               np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_8(self):
        npu_input1 = self.generate_single_data(-0.000000000000000000000000000000000000011754943508,
                                               -0.000000000000000000000000000000000000011754943508,
                                               ((2048, 31, 1, 2)),
                                               np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_float32_9(self):
        npu_input1 = self.generate_single_data(-0.000000000000000000000000000000000000011754943508,
                                               0.000000000000000000000000000000000000011754943508,
                                               ((8, 7, 149)),
                                               np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_out_float16_1(self):
        item = [[np.float16, 0, (8, 5, 132)], [np.float16, 0, (8, 5, 132)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 4)
        _, npu_output1 = create_common_tensor(item[1], 1, 4)
        cpu_output = self.cpu_op_exec_fp16(cpu_input1)
        npu_output = self.npu_out_op_exec(npu_input1, npu_output1)
        self.assertEqual(cpu_output.shape, npu_output.shape)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_cosh_out_float32_1(self):
        item = [[np.float32, 0, (8, 5, 132)], [np.float32, 0, (8, 5, 132)]]
        cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 4)
        _, npu_output1 = create_common_tensor(item[1], 1, 4)
        cpu_output = self.cpu_op_exec(cpu_input1)
        npu_output = self.npu_out_op_exec(npu_input1, npu_output1)
        self.assertEqual(cpu_output.shape, npu_output.shape)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
