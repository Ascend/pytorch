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

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestStd(TestCase):
    def cpu_op_exec(self, input, unbiased=True):
        output = torch.std(input, unbiased=unbiased)
        output = output.numpy()
        return output

    def npu_op_exec(self, input, unbiased=True):
        output = torch.std(input, unbiased=unbiased)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_dim_exec(self, input, dim, unbiased=True, keepdim=False):
        output = torch.std(input, dim, unbiased=unbiased, keepdim=keepdim)
        output = output.numpy()
        return output

    def npu_op_dim_exec(self, input, dim, unbiased=True, keepdim=False):
        output = torch.std(input, dim, unbiased=unbiased, keepdim=keepdim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_dim_out_exec(self, input, dim, output1, unbiased=True, keepdim=False):
        torch.std(input, dim, unbiased=unbiased, keepdim=keepdim,out=output1)
        output1 = output1.numpy()
        return output1

    def npu_op_dim_out_exec(self, input, dim, output1, unbiased=True, keepdim=False):
        torch.std(input, dim, unbiased=unbiased, keepdim=keepdim,out=output1)
        output1 = output1.to("cpu")
        output1 = output1.numpy()
        return output1
    
    def output_shape(self, inputshape, dim, unbiased=True, keepdim=False):
        shape = list(inputshape)
        if dim < len(inputshape) and keepdim == True:
            shape[dim] = 1
        elif dim < len(inputshape) and keepdim == False:
            shape.pop(dim)
        return shape
    
    def create_output_tensor(self, minvalue,maxvalue,shape,npuformat,dtype):
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input1)
        npu_input = torch.from_numpy(input1).npu()
        if npuformat != -1:
            npu_input = npu_input.npu_format_cast(npuformat)
        return cpu_input, npu_input
        
    def test_std_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [[16], [32, 1024], [32, 8, 1024], [128, 32, 8, 1023]]
        unbiased_list = [True, False]
        shape_format = [
            [np.float16, i, j, k] for i in format_list for j in shape_list for k in unbiased_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1 = self.cpu_op_exec(cpu_input1, item[3])
            cpu_output1 = cpu_output1.astype(np.float16)
            npu_output1 = self.npu_op_exec(npu_input1, item[3])
            self.assertRtolEqual(cpu_output1, npu_output1, prec16=0.1)

    def test_std_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        unbiased_list = [True, False]
        shape_format = [
            [np.float32, i, j, k] for i in format_list for j in shape_list for k in unbiased_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[3])
            npu_output = self.npu_op_exec(npu_input1, item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_std_dim_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1023]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        shape_format = [
            [np.float16, i, j, k, l, m] for i in format_list for j in shape_list 
            for k in dim_list for l in unbiased_list for m in keepdim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1 = self.cpu_op_dim_exec(cpu_input1, item[3], item[4], item[5])
            cpu_output1 = cpu_output1.astype(np.float16)
            npu_output1 = self.npu_op_dim_exec(npu_input1, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output1, npu_output1, prec16=0.003)

    def test_std_dim_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1023]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        shape_format = [
            [np.float32, i, j, k, l, m] for i in format_list for j in shape_list
            for k in dim_list for l in unbiased_list for m in keepdim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output1 = self.cpu_op_dim_exec(cpu_input1, item[3], item[4], item[5])
            npu_output1 = self.npu_op_dim_exec(npu_input1, item[3], item[4], item[5])
            self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_std_dim_out_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [[1024], [32, 24], [32, 8, 24], [12, 32, 8, 24]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        shape_format = [
            [np.float16, i, j, k, l, m] for i in format_list for j in shape_list
            for k in dim_list for l in unbiased_list for m in keepdim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            outputshape = self.output_shape(item[2],item[3],item[4],item[5])
            print(outputshape)
            print(item[2])
            cpu_output,npu_output = self.create_output_tensor(0,1,outputshape,item[1],item[0])
            if item[0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output1 = self.cpu_op_dim_out_exec(cpu_input1, item[3], cpu_output, item[4], item[5])
            npu_output1 = self.npu_op_dim_out_exec(npu_input1, item[3], npu_output, item[4], item[5])
            if item[0] == np.float16:
                cpu_output1 = cpu_output1.astype(np.float16)
            self.assertRtolEqual(cpu_output1, npu_output1, prec16=0.002)

    def test_std_dim_out_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [[1024], [32, 24], [32, 8, 24], [12, 32, 8, 24]]
        dim_list = [0]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        shape_format = [
            [np.float32, i, j, k, l, m] for i in format_list for j in shape_list
            for k in dim_list for l in unbiased_list for m in keepdim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            outputshape = self.output_shape(item[2],item[3],item[4],item[5])
            cpu_output,npu_output = self.create_output_tensor(0,1,outputshape,item[1],item[0])
            cpu_output1 = self.cpu_op_dim_out_exec(cpu_input1, item[3], cpu_output, item[4], item[5])
            npu_output1 = self.npu_op_dim_out_exec(npu_input1, item[3], npu_output, item[4], item[5])
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_std_dim_name_fp16(self, device):
        shape = (1024, 8, 32)
        cpu_input = torch.rand(shape, dtype=torch.float32)
        npu_input = cpu_input.npu().to(torch.float16)
        cpu_input.names = ['N','C','H']
        npu_input.names = ['N','C','H']
        dim = np.random.choice(['N', 'C', 'H'])
        cpu_output = torch.std(cpu_input, dim=dim)
        npu_output = torch.std(npu_input, dim=dim)
        self.assertRtolEqual(cpu_output.to(torch.float16).numpy(), npu_output.cpu().numpy())

    def test_std_dim_name_fp32(self, device):
        shape = (1024, 8, 32)
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        npu_input = cpu_input.npu()
        dim = np.random.choice(['N', 'C', 'H'])
        cpu_output = torch.std(cpu_input, dim=dim)
        npu_output = torch.std(npu_input, dim=dim)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_std_dim_out_name_fp16(self, device):
        shape = (1024, 8, 32)
        dimlist = ['N', 'C', 'H']
        cpu_input = torch.rand(shape, dtype=torch.float32)
        npu_input = cpu_input.npu()
        dim = np.random.choice(dimlist)
        dims = dimlist.index(dim)
        outputshape = self.output_shape(shape, dims)
        cpu_output,npu_output = self.create_output_tensor(0, 1, outputshape, -1, np.float32)
        npu_input = npu_input.to(torch.float16)
        npu_output = npu_output.to(torch.float16)
        cpu_input.names = ['N','C','H']
        npu_input.names = ['N','C','H']

        cpu_output = torch.std(cpu_input, dim=dim,out=cpu_output)
        npu_output = torch.std(npu_input, dim=dim,out=npu_output)
        cpu_output = cpu_output.to(torch.float16)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_std_dim_out_name_fp32(self, device):
        shape = (1024, 8, 32)
        dimlist = ['N', 'C', 'H']
        cpu_input = torch.rand(shape, dtype=torch.float32, names=('N', 'C', 'H'))
        npu_input = cpu_input.npu()
        dim = np.random.choice(dimlist)
        dims = dimlist.index(dim)
        outputshape = self.output_shape(shape, dims)
        cpu_output,npu_output = self.create_output_tensor(0, 1, outputshape, -1, np.float32)
        cpu_output = torch.std(cpu_input, dim=dim,out=cpu_output)
        npu_output = torch.std(npu_input, dim=dim,out=npu_output)
        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())
    
instantiate_device_type_tests(TestStd, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
