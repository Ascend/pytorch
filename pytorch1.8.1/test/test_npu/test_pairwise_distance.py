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

class TestPairwiseDistance(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def cpu_op_exec_default(self, input1, input2):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()
            input2 = input2.float()       
        pdist = torch.nn.PairwiseDistance()
        output = pdist(input1, input2)
        if stype == torch.float16:
            output = output.half()       
        output = output.numpy()
        return output
        
    def npu_op_exec_default(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        pdist = torch.nn.PairwiseDistance()
        output = pdist(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, input2, p, eps, keepdim):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()  
            input2 = input2.float()                  
        pdist = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        output = pdist(input1, input2)
        if stype == torch.float16:
            output = output.half()       
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, p, eps, keepdim):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        pdist = torch.nn.PairwiseDistance(p=p, eps=eps, keepdim=keepdim)
        output = pdist(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output
 
    def test_pairwise_distance_5_360_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (5, 360), np.float16)
        cpu_output = self.cpu_op_exec_default(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_default(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_pairwise_distance_10_3600_30_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (10, 3600, 30), np.float32)
        cpu_output = self.cpu_op_exec_default(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_default(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)         

    def test_pairwise_distance_505_10_30_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (505, 10, 30), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 0, 1e-6, True)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0, 1e-6, True)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_505_10_30_23_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (505, 10, 30, 23), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 0, 1e-5, True)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0, 1e-5, True)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_505_1_30_23_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (505, 1, 30, 23), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 0, 0, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0, 0, False)
        self.assertRtolEqual(cpu_output, npu_output) 


    def test_pairwise_distance_55_10_30_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (55, 10, 30), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 10, -1e-6, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 10, -1e-6, False)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_30_23_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (30, 23), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 10, 5, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 10, 5, False)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_505_1_23_19_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (505, 1, 23, 19), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 10, 10, True)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 10, 10, True)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_30_23_19_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (30, 23, 19), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 10, -10, True)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 10, -10, True)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_505_1_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (505, 1), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 0, -1e-4, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0, -1e-4, False)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_1_520_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (1, 520), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 10, 1e-4, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 10, 1e-4, False)
        self.assertRtolEqual(cpu_output, npu_output) 
 
    def test_pairwise_distance_1_1_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (1, 1), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 1,  2, True)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 1,  2, True)
        self.assertRtolEqual(cpu_output, npu_output) 

    #can't pass test
    def test_pairwise_distance_505_12_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (505, 12), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 0, -1e-4, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0, -1e-4, False)
        self.assertRtolEqual(cpu_output, npu_output) 

    def test_pairwise_distance_509_35_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, (509, 35), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, 0.0, -1, False)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.0, -1, False)
        self.assertRtolEqual(cpu_output, npu_output) 
instantiate_device_type_tests(TestPairwiseDistance, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()
