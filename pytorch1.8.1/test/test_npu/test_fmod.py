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


class TestFmod(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, input2):
        output = torch.fmod(input1, input2)
        # output = torch.fmod(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.fmod(input1, input2)
        # output = torch.fmod(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.fmod(input1, input2)
        # output = torch.fmod(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        # output = input1 + input2
        output = torch.fmod(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input3.to("npu")
        torch.fmod(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

            
    def test_fmod_scalar_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_fmod_uncontiguous_float32_scalar(self, device):
        def cpu_uncontiguous_op_exec_scalar(input1, input2):
            input1 = input1.as_strided([2, 2], [1, 2], 1)
            output = torch.fmod(input1, input2)
            output = output.numpy()
            return output

        def npu_uncontiguous_op_exec_scalar(input1, input2):
            input1 = input1.to("cpu")
            input1 = input1.as_strided([2, 2], [1, 2], 1)
            output = torch.fmod(input1, input2)
            output = output.to("cpu")
            output = output.numpy()
            return output

        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = cpu_uncontiguous_op_exec_scalar(cpu_input1, 2)
        npu_output = npu_uncontiguous_op_exec_scalar(npu_input1, 2)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestFmod, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:7")
    run_tests()