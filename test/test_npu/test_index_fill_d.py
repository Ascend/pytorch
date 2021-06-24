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


class TestIndexFillD(TestCase):

    def generate_x_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    # cpu
    def cpu_op_exec(self, x, dim, index, value):
        output = torch.index_fill(x, dim, index, value)
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, x, dim, index, value):
        x = x.to(torch.float32)
        output = torch.index_fill(x, dim, index, value)
        output = output.numpy().astype(np.float16)
        return output

    # torch.index_fill(input, dim, index, value) 
    # value is scalsr type or tensor type
    def npu_op_exec_interface1(self, x, dim, index, value):
        x = x.to("npu")
        index = index.to("npu")
        if type(value) == torch.Tensor:
            value = value.to("npu")
        output = torch.index_fill(x, dim, index, value)
        output = output.to("cpu").numpy()
        return output 
    # input.index_fill(dim, index, value)
    # value is scalsr type or tensor type
    def npu_op_exec_interface2(self, x, dim, index, value):
        x = x.to("npu")
        index = index.to("npu")
        if type(value) == torch.Tensor:
            value = value.to("npu")
        output = x.index_fill(dim, index, value)
        output = output.to("cpu").numpy()
        return output  

    # input.index_fill_(dim, index, value)
    # value is scalsr type or tensor type
    def npu_op_exec_interface3(self, x, dim, index, value):
        x = x.to("npu")
        index = index.to("npu")
        if type(value) == torch.Tensor:
            value = value.to("npu")
        x.index_fill_(dim, index, value)
        output = x.to("cpu").numpy()
        return output  

    def index_fill(self, testcases, value, dtype = "fp32"):
        for i, item in enumerate(testcases):
            index = torch.LongTensor(item[4])
            # testcase(s) for interface1
            npuinput_x1 = self.generate_x_data(item[0], item[1], item[2], item[5])
            if dtype == "fp16":
                cpu_output1_fp16 = self.cpu_op_exec_fp16(npuinput_x1, item[3], index, value)
                npu_output1 = self.npu_op_exec_interface1(npuinput_x1, item[3], index, value)
                self.assertRtolEqual(cpu_output1_fp16, npu_output1)
            else:
                cpu_output1 = self.cpu_op_exec(npuinput_x1, item[3], index, value)
                npu_output1 = self.npu_op_exec_interface1(npuinput_x1, item[3], index, value)
                self.assertRtolEqual(cpu_output1, npu_output1)

            # testcase(s) for interface2
            npuinput_x2 = self.generate_x_data(item[0], item[1], item[2], item[5])
            if dtype == "fp16":
                cpu_output2_fp16 = self.cpu_op_exec_fp16(npuinput_x2, item[3], index, value)
                npu_output2 = self.npu_op_exec_interface2(npuinput_x2, item[3], index, value)
                self.assertRtolEqual(cpu_output2_fp16, npu_output2)
            else:
                cpu_output2 = self.cpu_op_exec(npuinput_x2, item[3], index, value)
                npu_output2 = self.npu_op_exec_interface2(npuinput_x2, item[3], index, value)
                self.assertRtolEqual(cpu_output2, npu_output2)

             # testcase(s) for interface2
            npuinput_x3 = self.generate_x_data(item[0], item[1], item[2], item[5])
            if dtype == "fp16":
                cpu_output3_fp16 = self.cpu_op_exec_fp16(npuinput_x3, item[3], index, value)
                npu_output3 = self.npu_op_exec_interface3(npuinput_x3, item[3], index, value)
                self.assertRtolEqual(cpu_output3_fp16, npu_output3)
            else:
                cpu_output3 = self.cpu_op_exec(npuinput_x3, item[3], index, value)
                npu_output3 = self.npu_op_exec_interface3(npuinput_x3, item[3], index, value)
                self.assertRtolEqual(cpu_output3, npu_output3)


    #pylint: disable=unused-argument
    def test_index_fill_d(self, device):
        
        testcases = [  #minV, maxV, shape,   dim, index,   dtype
                      # fp32
                      [-10, 10, (2, 2, 3, 0), 1, [0, 1], np.float32], # spical case 
                      [-10, 10, (2, 2, 3, 3), 1, [0, 1], np.float32],
                      [-10, 10, (2,), 0, [0, 1], np.float32],
                      [-100, 100, (2, 4, 6, 8, 10, 12), 0, [0, 1], np.float32],
                      [-0.000030517578125, 0.000030517578125, (2,32,149,31), 0, [0, 1], np.float32],
                      [-3402823500.0, 3402823500.0, (2,32,149,31), 0, [0, 1], np.float32],
                      [-100, 100, (65535, 2, 2, 2, 2, 2), 0, [0, 1, 10, 20], np.float32],
                      [-100, 100, (2, 65535, 2, 2, 2, 2), 0, [0, 1], np.float32],
                      [-100, 100, (2, 2, 65535, 2, 2, 2), 0, [0, 1], np.float32],
                      [-100, 100, (2, 2, 2, 65535, 2, 2), 0, [0, 1], np.float32],
                      [-100, 100, (2, 2, 2, 2, 65535, 2), 0, [0, 1], np.float32],
                      [-100, 100, (2, 2, 2, 2, 2, 65535), 0, [0, 1], np.float32],
                      # int32
                      [-10, 10, (2, 2, 3, 0), 1, [0, 1], np.int32], # spical case 
                      [-10, 10, (2, 2, 3, 3), 1, [0, 1], np.int32],
                      [-10, 10, (2,), 0, [0, 1], np.int32],
                      [-100, 100, (2, 4, 6, 8, 10, 12), 0, [0, 1], np.int32],
                      [-3402823500, 3402823500, (2,32,149,31), 0, [0, 1], np.int32],
                      [-100, 100, (65535, 2, 2, 2, 2, 2), 0, [0, 1, 10, 20], np.int32],
                      [-100, 100, (2, 65535, 2, 2, 2, 2), 0, [0, 1], np.int32],
                      [-100, 100, (2, 2, 65535, 2, 2, 2), 0, [0, 1], np.int32],
                      [-100, 100, (2, 2, 2, 65535, 2, 2), 0, [0, 1], np.int32],
                      [-100, 100, (2, 2, 2, 2, 65535, 2), 0, [0, 1], np.int32],
                      [-100, 100, (2, 2, 2, 2, 2, 65535), 0, [0, 1], np.int32],
                   ]

        testcases_fp16 = [  #minV, maxV, shape,   dim, index,   dtype
                      # fp16
                      [-10, 10, (2, 2, 3, 3), 1, [0, 1], np.float16],
                      [-10, 10, (2,), 0, [0, 1], np.float16],
                      [-100, 100, (2, 4, 6, 8, 10, 12), 0, [0, 1], np.float16],
                      [-60000, 60000, (2,32,149,31), 0, [0, 1], np.float16],
                      [-100, 100, (65535, 2, 2, 2, 2, 2), 0, [0, 1, 10, 20], np.float16],
                      [-100, 100, (2, 65535, 2, 2, 2, 2), 0, [0, 1], np.float16],
                      [-100, 100, (2, 2, 65535, 2, 2, 2), 0, [0, 1], np.float16],
                      [-100, 100, (2, 2, 2, 65535, 2, 2), 0, [0, 1], np.float16],
                      [-100, 100, (2, 2, 2, 2, 65535, 2), 0, [0, 1], np.float16],
                      [-100, 100, (2, 2, 2, 2, 2, 65535), 0, [0, 1], np.float16],
                   ]

        # Test three interfaces for fp32, int32, fp16 with scalar value.
        # Example.
        # input  = torch.randn(3, 3, 4)
        # index = torch.LongTensor([1, 2])
        # value  = 5
        # 1. output = torch.index_fill(input, dim, index, value)  Out-Place
        # 2. output = input.index_fill(dim, index, value)  Out-Place
        # 3. inpue.index_fill_(dim, index, value)  In-Place

        value = np.random.uniform(-10000, 10000)
        self.index_fill(testcases=testcases, value=value)
        self.index_fill(testcases=testcases_fp16, value=value, dtype="fp16")

        # Test three interfaces for fp32, int32, fp16 with tensor value.
        # Example.
        # input  = torch.randn(3, 3, 4)
        # index = torch.LongTensor([1, 2])
        # value  = torch.tensor(5)
        # 1. output = torch.index_fill(input, dim, index, value)  Out-Place
        # 2. output = input.index_fill(dim, index, value)  Out-Place
        # 3. inpue.index_fill_(dim, index, value)  In-Place
        value_tensor = torch.tensor(value)
        self.index_fill(testcases=testcases, value=value_tensor)
        self.index_fill(testcases=testcases_fp16, value=value_tensor, dtype="fp16")

instantiate_device_type_tests(TestIndexFillD, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
