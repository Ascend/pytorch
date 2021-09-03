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
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests
import time
import os

class NetFirst(torch.nn.Module):
    def __init__(self):
        super(NetFirst, self).__init__()
        self.cont = 0

    def forward(self):
        self.cont += 1
        return self.cont
        
class NetStatic(torch.nn.Module):
    def __init__(self):
        super(NetStatic, self).__init__()

    def forward(self, x):
        out = torch.mul(x, x)  
        return out
        
class NetDynamic(torch.nn.Module):
    def __init__(self):
        super(NetDynamic, self).__init__()

    def forward(self, x):
        out1 = torch.mul(x, x)
        out2 = torch.relu(out1)
        out3 = torch.neg(out2)
        out4 = torch.floor_divide(x, out2)
        out5 = torch.div(out4, out3)
        out6 = torch.mul(out5, 0.8)
        return out6
        
class NetUnsupport(torch.nn.Module):
    def __init__(self):
        super(NetUnsupport, self).__init__()

    def forward(self, x):
        out1 = torch.sin(x)
        out2 = torch.sinh(out1)
        out3 = torch.selu(out2)
        return out3
        
class NetMixOp(torch.nn.Module):
    def __init__(self):
        super(NetMixOp, self).__init__()

    def forward(self, x):
        out0 = torch.floor_divide(x, x)
        out1 = torch.selu(out0)
        out2 = torch.sub(out1, x)
        out3 = torch.div(out2, x)
        out4 = torch.sin(out3)
        out5 = torch.mul(out4, x)
        out6 = torch.sinh(out5)
        out7 = torch.neg(out6)
        out8 = torch.relu(out7)     
        return out8
        

class TestShape(TestCase):
    def create_random_shape_tensor(self, item, minValue, maxValue):
        format = item[0]
        dtype = item[1]
        dim = item[2]
        shape = np.random.randint(1, 100, dim)
        input = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input)
        npu_input = torch.from_numpy(input).to("npu:0")
        if format not in (-1, 0):
            npu_input = npu_input.npu_format_cast(format)
        return cpu_input, npu_input
        
    def test_dynamic_first_step(self, device):
        net = NetFirst()
        net = net.to("npu")      
        for step in range(100):
            cpu_output = net()    
            npu_output = net()
            assert cpu_output == npu_output - 1
        
    def test_dynamic_static_shape(self, device):
        net = NetStatic()
        net = net.to("npu")
        item = [np.float32, 3, (10, 255, 5, 5)]   
        for step in range(100):
            cpu_tensor, npu_tensor = create_common_tensor(item, -100, 100)
            cpu_output = net(cpu_tensor)
            cpu_output = cpu_output.numpy()
            npu_output = net(npu_tensor)
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)
        
    def test_dynamic_op(self, device):
        net = NetDynamic()
        net = net.to("npu")
        shape_format = [
            [3, np.float32, 1],
            [3, np.float32, 2],
            [3, np.float32, 4]  
        ]       
        for step in range(100):     
            for item in shape_format:
                cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -100, 100)
                cpu_output = net(cpu_tensor)
                cpu_output = cpu_output.numpy()
                npu_output = net(npu_tensor)
                npu_output = npu_output.to("cpu")
                npu_output = npu_output.numpy()
                self.assertRtolEqual(cpu_output, npu_output)
          
    def test_dyamic_unspport_op(self, device):
        net = NetUnsupport()
        net = net.to("npu")
        item = [3, np.float32, 4]     
        for step in range(100):
            cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -100, 100)
            cpu_output = net(cpu_tensor)
            cpu_output = cpu_output.numpy()
            npu_output = net(npu_tensor)
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)
        
    def test_dynamic_mix_op(self, device):
        net = NetMixOp()
        net = net.to("npu")
        item = [3, np.float32, 4]       
        for step in range(100):
            cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -100, 100)
            cpu_output = net(cpu_tensor)
            cpu_output = cpu_output.numpy()
            npu_output = net(npu_tensor)
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)
        
    def test_dynamic_all_random_mix_op(self, device):
        net = NetMixOp()
        net = net.to("npu")
        format_list = [0, 3, 29]
        dtype_list = [np.float32]
        dim_list = [1, 2, 3, 4]
        items = [
            [i, j, k, 10] for i in format_list for j in dtype_list for k in dim_list
        ]
        for step in range(100):
            for item in items:
                cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -100, 100)
                cpu_output = net(cpu_tensor)
                cpu_output = cpu_output.numpy()
                npu_output = net(npu_tensor)
                npu_output = npu_output.to("cpu")
                npu_output = npu_output.numpy()
                self.assertRtolEqual(cpu_output, npu_output)
   
    def test_dynamic_exit(self, device):
        net = NetMixOp()
        net = net.to("npu")
        item = [3, np.float32, 4]
        for step in range(2):
            cpu_tensor, npu_tensor = self.create_random_shape_tensor(item, -100, 100)
            cpu_output = net(cpu_tensor)
            cpu_output = cpu_output.numpy()
            npu_output = net(npu_tensor)
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            self.assertRtolEqual(cpu_output, npu_output)
         
    
instantiate_device_type_tests(TestShape, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()  
