# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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


class Testisin(TestCase):

    @staticmethod
    def generate_data(min_d, max_d, shape, dtype, scalar_type='float'):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        if scalar_type.startswith('float'):
            npu_input2 = np.random.uniform(min_d, max_d)
        else:
            npu_input2 = np.random.randint(min_d, max_d)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1, npu_input2

    @staticmethod
    def cpu_op_exec(input1, input2):
        output = torch.isin(input1, input2)
        output = output.numpy()
        return output

    @staticmethod
    def cpu_op_exec_assume_unique_invert(input1, input2, assume_unique, invert):
        output = torch.isin(input1, input2, assume_unique=assume_unique, invert=invert)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec_tensor_need_to_npu(input1, input2):
        input1 = input1.to("npu")
        output = torch.isin(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec_tensor_need_to_npu_assume_unique_invert(input1, input2, assume_unique, invert):
        input1 = input1.to("npu")
        output = torch.isin(input1, input2, assume_unique=assume_unique, invert=invert)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_isin_int(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.int32, scalar_type='int')
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isin_float(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4,), np.float, scalar_type='float')
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isin_int_float(self):
        npu_input1, npu_input2 = self.generate_data(-100, 100, (4, 3, 2), np.int32, scalar_type='float')
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isin_float_int(self):
        npu_input1, npu_input2 = self.generate_data(10, 20, (4, 3), np.float32, scalar_type='int')
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isin_invert_false(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3, 2), np.float32)
        assume_unique = False
        invert = False
        cpu_output = self.cpu_op_exec_assume_unique_invert(
            npu_input1, npu_input2, assume_unique=assume_unique, invert=invert)
        npu_output = self.npu_op_exec_tensor_need_to_npu_assume_unique_invert(
            npu_input1, npu_input2, assume_unique=assume_unique, invert=invert)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isin_invert_true(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3, 2), np.float32)
        assume_unique = False
        invert = True
        cpu_output = self.cpu_op_exec_assume_unique_invert(
            npu_input1, npu_input2, assume_unique=assume_unique, invert=invert)
        npu_output = self.npu_op_exec_tensor_need_to_npu_assume_unique_invert(
            npu_input1, npu_input2, assume_unique=assume_unique, invert=invert)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
