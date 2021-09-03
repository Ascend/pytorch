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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestIndexPut(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    # cpu
    def cpu_op_exec(self, input_x, index, value, accumulate):
        output = torch.index_put(input_x, index, value, accumulate)
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input_x, index, value, accumulate):
        input_x = input_x.to(torch.float32)
        output = torch.index_put(input_x, index, value, accumulate)
        output = output.numpy().astype(np.float16)
        return output

    def cpu_op_exec_interface(self, input_x, index, value):
        input_x[index] = value
        output = input_x
        output = output.numpy() 
        return output

    # npu
    def npu_op_exec_interface1(self, input_x, index, value, accumulate):
        input_x = input_x.to("npu")
        index_npu1 = index[0].to("npu")
        index_npu2 = index[1].to("npu")
        index_npu = (index_npu1, index_npu2)
        if type(value) == torch.Tensor:
            value = value.to("npu")
        output = input_x.index_put(index_npu, value, accumulate)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_interface2(self, input_x, index, value, accumulate):
        input_x = input_x.to("npu")
        index_npu1 = index[0].to("npu")
        index_npu2 = index[1].to("npu")
        index_npu = (index_npu1, index_npu2)
        if type(value) == torch.Tensor:
            value = value.to("npu")
        output = input_x.index_put_(index_npu, value, accumulate)
        output = output.to("cpu")
        output = output.numpy() 
        return output
    
    def npu_op_exec_interface3(self, input_x, index, value):
        input_x = input_x.to("npu")
        index_npu1 = index[0].to("npu")
        index_npu2 = index[1].to("npu")
        index_npu = (index_npu1, index_npu2)
        if type(value) == torch.Tensor:
            value = value.to("npu")
        input_x[index_npu] = value
        output = input_x
        output = output.to("cpu")
        output = output.numpy()
        return output
            
    # assertRtolEqual
    def index_put(self, testcases, value, dtype = "fp32"):
        for i, item in enumerate(testcases):
            index = (torch.LongTensor(item[4][0]), torch.LongTensor(item[4][1]))
            #test for index_put
            npuinput_x1 = self.generate_data(item[0], item[1], item[2], item[5])
            if dtype == "fp16":
                cpu_output1 = self.cpu_op_exec_fp16(npuinput_x1, index, value, item[3])
                npu_output1 = self.npu_op_exec_interface1(npuinput_x1, index, value, item[3])
                self.assertRtolEqual(cpu_output1, npu_output1)
            else:
                cpu_output1 = self.cpu_op_exec(npuinput_x1, index, value, item[3])
                npu_output1 = self.npu_op_exec_interface1(npuinput_x1, index, value, item[3])
                self.assertRtolEqual(cpu_output1, npu_output1)

            #test for index_put_
            npuinput_x2 = self.generate_data(item[0], item[1], item[2], item[5])
            if dtype == "fp16":
                cpu_output2 = self.cpu_op_exec_fp16(npuinput_x2, index, value, item[3])
                npu_output2 = self.npu_op_exec_interface2(npuinput_x2, index, value, item[3])
                self.assertRtolEqual(cpu_output2, npu_output2)
            else:
                cpu_output2 = self.cpu_op_exec(npuinput_x2, index, value, item[3])
                npu_output2 = self.npu_op_exec_interface2(npuinput_x2, index, value, item[3])
                self.assertRtolEqual(cpu_output2, npu_output2)
            
            #test for input[index] = value
            npuinput_x3 = self.generate_data(item[0], item[1], item[2], item[5])
            if dtype == "fp16":
                cpu_output3 = self.cpu_op_exec_interface(npuinput_x3.to(torch.float32), index, value)
                npu_output3 = self.npu_op_exec_interface3(npuinput_x3, index, value)
                self.assertRtolEqual(cpu_output3.astype(npu_output3.dtype), npu_output3)
            else:
                cpu_output3 = self.cpu_op_exec_interface(npuinput_x3, index, value)
                npu_output3 = self.npu_op_exec_interface3(npuinput_x3, index, value)
                self.assertRtolEqual(cpu_output3, npu_output3)
    
    def test_index_put_d(self, device):
        testcases_fp32 = [  
                #minV, maxV, shape, accumulate, index, dtype        
                # fp32
                #IndexPut_fp32_accumulate1_001
                [-10, 10, (2, 2, 3, 3), True, [[1,1],[0,1]], np.float32],
                
                # IndexPut_fp32_accumulate0_002      
                [-10, 10, (2, 2, 3, 3), False, [[1,1],[0,1]], np.float32],
                
                #IndexPut_fp32_accumulate1_003
                [-100, 100, (2, 4, 6, 8, 10, 12), True, [[1,1],[0,1]], np.float32],
                
                #IndexPut_fp32_accumulate0_004
                [-100, 100, (2, 4, 6, 8, 10, 12), False, [[1,1],[0,1]], np.float32],
                
                #IndexPut_fp32_accumulate1_R0.5e16_005
                [-0.000030517578125, 0.000030517578125, (2,32,149,31), True, [[1,1],[0,1]], np.float32],
                
                #IndexPut_fp32_accumulate0_R0.5e16_006
                [-0.000030517578125, 0.000030517578125, (2,32,149,31), False,[[1,1],[0,1]], np.float32],
                
                #IndexPut_fp32_accumulate1_R2e32_007
                [-3402823500.0, 3402823500.0, (2,32,149,31), True, [[1,1],[0,1]], np.float32],

                #IndexPut_fp32_accumulate0_R2e32_008
                [-3402823500.0, 3402823500.0, (2,32,149,31), False, [[1,1],[0,1]], np.float32],

                #IndexPut_fp32_accumulate1_S2e16_009
                [-100, 100, (65535, 2, 2, 2, 2, 2), True, [[1,1],[0,1]], np.float32],

                #IndexPut_fp32_accumulate0_S2e16_010
                [-100, 100, (65535, 2, 2, 2, 2, 2), False, [[1,1],[0,1]], np.float32],
                
                ]
        testcases_fp16 = [
                #IndexPut_fp16_accumulate1_011
                [-10, 10, (2, 2, 3, 3), True, [[1,1],[0,1]], np.float16],

                #IndexPut_fp16_accumulate0_012
                [-10, 10, (2, 2, 3, 3), False, [[1, 1],[0,1]], np.float16],
                
                #IndexPut_fp16_accumulate1_013
                [-100, 100, (2, 4, 6, 8, 10, 12), True, [[1,1],[0,1]], np.float16],
                
                #IndexPut_fp16_accumulate0_014
                [-100, 100, (2, 4, 6, 8, 10, 12), False, [[1,1],[0,1]], np.float16],
                
                #IndexPut_fp16_accumulate1_R2e16_015
                [-60000,60000, (2,32,149,31), True, [[1,1],[0,1]], np.float16],
                
                #IndexPut_fp16_accumulate0_R2e16_016
                [-60000,60000, (2,32,149,31), True, [[1,1],[0,1]], np.float16],
                
                #IndexPut_fp16_accumulate1_S2e16_017
                [-100, 100, (65535, 2, 2, 2, 2, 2), True, [[1,1],[0,1]], np.float16],
                
                #IndexPut_fp16_accumulate0_S2e16_018
                [-100, 100, (65535, 2, 2, 2, 2, 2), False, [[1,1],[0,1]], np.float16],                
                ]
        value = np.random.uniform(-10000, 10000)
        value_tensor = torch.tensor(value)
        self.index_put(testcases=testcases_fp32, value=value_tensor)
        self.index_put(testcases=testcases_fp16, value=value_tensor, dtype="fp16")
        
instantiate_device_type_tests(TestIndexPut, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()
    