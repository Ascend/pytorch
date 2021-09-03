#  Copyright (c) 2020, Huawei Technologies.All rights reserved.
#  Licensed under the BSD 3-Clause License  (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://opensource.org/licenses/BSD-3-Clause
#
#  Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestTriangularSolve(TestCase):
    def generate_data(self, min, max, shape, dtype): 
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1) 
        return npu_input1

    def cpu_op_exec(self, input1, input2, input3, input4, input5): 
        output = input1.triangular_solve(input2,upper=input3,transpose=input4,unitriangular=input5)
        return output 

    def cpu_op_exec_float16(self, input1, input2, input3, input4, input5): 
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = input1.triangular_solve(input2,upper=input3,transpose=input4,unitriangular=input5)
        return output 

    def npu_op_exec(self, input1, input2, input3, input4, input5): 
        input1 = input1.to("npu") 
        input2 = input2.to("npu") 
        output = input1.triangular_solve(input2,upper=input3,transpose=input4,unitriangular=input5)
        output = output.to("cpu") 
        return output        

    def test_triangular_solve_float32(self, device): 
        npu_input1 = self.generate_data(0, 100, (2,3) , np.float32) 
        npu_input2 = self.generate_data(0, 100, (2,2) , np.float32) 
        npu_true = True 
        npu_false = False
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_true, npu_false, npu_false) 
        #npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_true, npu_false, npu_false) 
        #self.assertRtolEqual(cpu_output, npu_output) 

    def test_triangular_solve_float32_zhuanzhi(self, device): 
        npu_input1 = self.generate_data(0, 100, (2,3) , np.float32) 
        npu_input2 = self.generate_data(0, 100, (2,2) , np.float32) 
        npu_true = True 
        npu_false = False
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_true, npu_true, npu_false) 
        #npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_true, npu_true, npu_false) 
        #self.assertRtolEqual(cpu_output, npu_output) 

    def test_triangular_solve_float32_danwei(self, device): 
        npu_input1 = self.generate_data(0, 100, (2,3) , np.float32) 
        npu_input2 = self.generate_data(0, 100, (2,2) , np.float32) 
        npu_true = True 
        npu_false = False
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_true, npu_false, npu_true) 
        #npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_true, npu_false, npu_true) 
        #self.assertRtolEqual(cpu_output, npu_output) 

    def test_triangular_solve_float16(self, device): 
        npu_input1 = self.generate_data(0, 100, (2,3) , np.float16) 
        npu_input2 = self.generate_data(0, 100, (2,2) , np.float16) 
        npu_true = True 
        npu_false = False
        cpu_output = self.cpu_op_exec_float16(npu_input1, npu_input2, npu_true, npu_false, npu_true) 
        #npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_true, npu_false, npu_true) 
        #self.assertRtolEqual(cpu_output, npu_output) 

instantiate_device_type_tests(TestTriangularSolve, globals(), except_for='cpu')
if __name__ == '__main__':
    torch.npu.set_device("npu:2") 
    run_tests()
