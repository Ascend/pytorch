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

class TestReplicationPad2d(TestCase):    
    def cpu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad2d(pad)
        output = m(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, pad):
        m = torch.nn.ReplicationPad2d(pad).to("npu")      
        output = m(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_replicationPad2d_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (1, 1, 2,3 )],[2,2,2,2]],
                [[np.float32, 3, (1, 1, 4,3 )],2]
                ] 
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_replicationPad2d_shape_format_fp16(self, device):
        shape_format = [
                [[np.float16, 0, (1, 1, 4,3 )],[2,2,2,2]],
                [[np.float16, 3, (1, 1, 4,3 )],3]
                ]
        def cpu_op_exec_fp16(input1, pad):
            input1 = input1.to(torch.float32)
            m = torch.nn.ReplicationPad2d(pad)
            output = m(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_replicationPad2d_shape_format_int8(self, device):
        shape_format = [
                [[np.int8, 0, (1, 1, 4,3 )],[2,2,2,2]],
                [[np.int8, 0, (1, 1, 5,3 )],6]
                ]
        def cpu_op_exec_int8(input1, pad):
            input1 = input1.to(torch.float32)
            m = torch.nn.ReplicationPad2d(pad)
            output = m(input1)
            output = output.numpy()
            output = output.astype(np.int8)
            return output
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_int8(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_replicationPad2d_shape_format_uint8(self, device):
        shape_format = [
                [[np.uint8, 0, (1, 1, 4,3 )],[2,2,2,2]],
                [[np.uint8, 0, (1, 1, 4, 9 )],2]
                ]
        def cpu_op_exec_uint8(input1, pad):
            input1 = input1.to(torch.float32)
            m = torch.nn.ReplicationPad2d(pad)
            output = m(input1)
            output = output.numpy()
            output = output.astype(np.uint8)
            return output

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_uint8(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_replicationPad2d_shape_format_int32(self, device):
        shape_format = [
                [[np.int32, 0, (1, 1, 4,3 )],[2,2,2,2]],
                [[np.int32, 0, (1, 1, 4, 9 )],2]
                ]
        def cpu_op_exec_int32(input1, pad):
            input1 = input1.to(torch.float32)
            m = torch.nn.ReplicationPad2d(pad)
            output = m(input1)
            output = output.numpy()
            output = output.astype(np.int32)
            return output

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_int32(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestReplicationPad2d, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
