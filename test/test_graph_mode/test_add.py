# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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
from torch_npu.testing.decorator import instantiate_tests, graph_mode
from torch_npu.testing.common_utils import create_common_tensor

@instantiate_tests
class TestAdd(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = input1 + input2
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input1 + input2
        output = output.to("cpu")
        output = output.numpy()
        return output    

    @graph_mode
    def test_add_common_shape_format(self):
        shape_format = [
                [[np.float32, -1, (4, 3)],    [np.float32, -1, (4, 3)]],
                [[np.float32, -1, (4, 3, 1)], [np.float32, -1, (4, 1, 5)]],
                [[np.int32,   -1, (2, 3)],    [np.int32,   -1, (2, 3)]],
                [[np.int32,   -1, (4, 3, 1)], [np.int32,   -1, (4, 1, 5)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)  

    @graph_mode
    def test_add_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = input1 + input2
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
                [[np.float16, -1, (4, 3)],    [np.float16, -1, (4, 3)]],
                [[np.float16, -1, (4, 3, 1)], [np.float16, -1, (4, 1, 5)]],
        ] 

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)  

    @graph_mode
    def test_add_float32_out(self):
        def npu_op_exec_out(input1, input2, input3):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            output = input3.to("npu")
            torch.add(input1, input2, out=output)
            output = output.to("cpu")
            output = output.numpy()
            return output 
        
        shape_format = [
                [[np.float16, -1, (4, 3)],    [np.float16, -1, (4, 3)],    [np.float16, -1, (4, 3)]],
                [[np.float16, -1, (4, 3, 1)], [np.float16, -1, (4, 1, 5)], [np.float16, -1, (4, 1, 5)]],
        ]
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)

    '''
    currently do not support scalar dtype
    @graph_mode
    def test_add_scalar_float32(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2,3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        self.assertRtolEqual(cpu_output, npu_output)
    '''

if __name__ == "__main__":
    run_tests()
