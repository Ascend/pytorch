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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestClamp(TestCase):

    def cpu_op_exec(self, input, min_val, max_val):
        output = torch.clamp(input, min_val, max_val)
        output = output.numpy()      
        return output 

    def cpu_inp_op_exec(self, input, min_val, max_val):
        output = torch.clamp_(input, min_val, max_val)
        output = output.numpy()
        return output       

    def cpu_inp_out_op_exec(self, input, min_val, max_val, output):        
        torch.clamp(input, min_val, max_val, out = output)
        output = output.numpy()
        return output
        
    def npu_op_exec(self, input, min_val, max_val):
        output = torch.clamp(input, min_val, max_val)
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def npu_inp_op_exec(self, input, min_val, max_val):
        torch.clamp_(input, min_val, max_val)
        output = input.to("cpu")
        output = output.numpy()
        return output

    def npu_inp_out_op_exec(self, input, min_val, max_val, output):
        torch.clamp(input, min_val, max_val, out = output)
        output = output.to("cpu")
        output = output.numpy()
        return output
            
    def test_clamp_x_shape_format_fp32(self, device):
        shape_format = [
                [[np.float32, 0, 1]],
                [[np.float32, 0, (64, 10)]],
                [[np.float32, 3, (256, 2048, 7, 7)]],
                [[np.float32, 4, (32, 1, 3, 3)]],
                [[np.float32, 29, (10, 128)]]                
        ]        
        for item in shape_format:
            cpu_input1, npu_input1 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_input2, npu_input2 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_input3, npu_input3 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_out, npu_out = create_common_tensor(item[0], -0.5, 0.5)
            cpu_output = self.cpu_op_exec(cpu_input1, -0.5, 0.5)
            npu_output = self.npu_op_exec(npu_input1, -0.5, 0.5)           
            cpu_inp_output = self.cpu_inp_op_exec(cpu_input2,-0.5, 0.5)
            npu_inp_output = self.npu_inp_op_exec(cpu_input2, -0.5, 0.5)
            cpu_inp_uncon_output = self.cpu_inp_out_op_exec(cpu_input3, -0.5, 0.5, cpu_out)
            npu_inp_uncon_output = self.npu_inp_out_op_exec(npu_input3, -0.5, 0.5, npu_out)           
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_x_shape_format_int32(self, device):
        shape_format = [
                [[np.int32, -1, 1]],
                [[np.int32, -1, (64, 10)]],
                [[np.int32, -1, (256, 2048, 7, 7)]],
                [[np.int32, -1, (32, 1, 3, 3)]],
                [[np.int32, -1, (10, 128)]],               
        ]        
        for item in shape_format:
            cpu_input1, npu_input1 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_input2, npu_input2 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_input3, npu_input3 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_out, npu_out = create_common_tensor(item[0], -0.5, 0.5)
            cpu_output = self.cpu_op_exec(cpu_input1, -2, 2)
            npu_output = self.npu_op_exec(npu_input1, -2, 2)           
            cpu_inp_output = self.cpu_inp_op_exec(cpu_input2, -2, 2)
            npu_inp_output = self.npu_inp_op_exec(npu_input2, -2, 2)
            cpu_inp_uncon_output = self.cpu_inp_out_op_exec(cpu_input3, -0.5, 0.5, cpu_out)
            npu_inp_uncon_output = self.npu_inp_out_op_exec(npu_input3, -0.5, 0.5, npu_out)          
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_x_shape_format_fp16(self, device):
        shape_format = [
                [[np.float16, 0, 1]],
                [[np.float16, 0, (64, 10)]],
                [[np.float16, 3, (256, 2048, 7, 7)]],
                [[np.float16, 4, (32, 1, 3, 3)]],
                [[np.float16, 29, (10, 128)]]                
        ]        
        for item in shape_format:
            cpu_input1, npu_input1 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_input2, npu_input2 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_input3, npu_input3 =  create_common_tensor(item[0], -0.5, 0.5)
            cpu_out, npu_out = create_common_tensor(item[0], -0.5, 0.5)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input3 = cpu_input3.to(torch.float32)
            cpu_out = cpu_out.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, -0.5, 0.5)
            npu_output = self.npu_op_exec(npu_input1, -0.5, 0.5)           
            cpu_inp_output = self.cpu_inp_op_exec(cpu_input2,-0.5, 0.5)
            npu_inp_output = self.npu_inp_op_exec(npu_input2, -0.5, 0.5)
            cpu_inp_uncon_output = self.cpu_inp_out_op_exec(cpu_input3, -0.5, 0.5, cpu_out)
            npu_inp_uncon_output = self.npu_inp_out_op_exec(npu_input3, -0.5, 0.5, npu_out)
            cpu_output = cpu_output.astype(np.float16)
            cpu_inp_output = cpu_inp_output.astype(np.float16)
            cpu_inp_uncon_output = cpu_inp_uncon_output.astype(np.float16)         
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

instantiate_device_type_tests(TestClamp, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
