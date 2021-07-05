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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestPdist(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec_default(self, input1):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()       
        output = torch.nn.functional.pdist(input1)
        if stype == torch.float16:
            output = output.half()       
        output = output.numpy()
        return output
        
    def npu_op_exec_default(self, input1):
        input1 = input1.to("npu")
        output = torch.nn.functional.pdist(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, p):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()       
        output = torch.nn.functional.pdist(input1, p)
        if stype == torch.float16:
            output = output.half()       
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, p):
        input1 = input1.to("npu")
        output = torch.nn.functional.pdist(input1, p)
        output = output.to("cpu")
        output = output.numpy()
        return output
 
    def test_pdist__5_360_float16(self, device):
        npu_input1 = self.generate_data(-2, 2, (5, 360), np.float16)
        cpu_output = self.cpu_op_exec_default(npu_input1)
        npu_output = self.npu_op_exec_default(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_10_3600_float32(self, device):
        npu_input1 =self.generate_data(-2, 2, (10, 3600), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1)
        npu_output = self.npu_op_exec_default(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)       

    def test_pdist_30_301_0_float16(self, device):
        npu_input1 = self.generate_data(-2, 2, (30, 301), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 0.0)
        npu_output = self.npu_op_exec(npu_input1, 0.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_10_256_0_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (10, 256), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 0.0)
        npu_output = self.npu_op_exec(npu_input1, 0.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_20_234_1_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (20, 234), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 1.0)
        npu_output = self.npu_op_exec(npu_input1, 1.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_10_1600_1_float16(self, device):
        npu_input1 = self.generate_data(-2, 2, (10, 1600), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 1.0)
        npu_output = self.npu_op_exec(npu_input1, 1.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_8_1025_2_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (8, 1025), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)               

    def test_pdist_9_10250_2_float16(self, device):
        npu_input1 = self.generate_data(-2, 2, (9, 10250), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 2.0)
        npu_output = self.npu_op_exec(npu_input1, 2.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist_100_7025_10_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (100, 7025), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 10.0)
        npu_output = self.npu_op_exec(npu_input1, 10.0)
        self.assertRtolEqual(cpu_output, npu_output)               

    def test_pdist_111_10025_10_float16(self, device):
        npu_input1 = self.generate_data(-2, 2, (111, 10025), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 10.0)
        npu_output = self.npu_op_exec(npu_input1, 10.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist__50_0_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (50, 0), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1)
        npu_output = self.npu_op_exec_default(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist__1_110_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (1, 110), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1)
        npu_output = self.npu_op_exec_default(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pdist__0_0_float32(self, device):
        npu_input1 = self.generate_data(-2, 2, (0, 0), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1)
        npu_output = self.npu_op_exec_default(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)         

instantiate_device_type_tests(TestPdist, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()