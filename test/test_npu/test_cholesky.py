import torch 
import numpy as np 
import sys 
import copy 
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import random
import math
 
class TestCholesky(TestCase):    
# pylint: disable=unused-variable,unused-argument
# pylint: disable=W,C
    def create_2d_tensor(self, item, minValue, maxValue):
        dtype = item[0]
        format = item[1]
        shape = item[2]
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        a = torch.from_numpy(input1)
        cpu_input = torch.matmul(a, a.t())
        npu_input = torch.matmul(a, a.t()).to("npu")
        if format != -1:
            npu_input = npu_input.npu_format_cast(format)
        return cpu_input, npu_input

    def create_nd_tensor(self, item, minValue, maxValue):
        dtype = item[0]
        format = item[1]
        shape = item[2]
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        a = torch.from_numpy(input1)
        a = a.to(torch.float32)
        cpu_input = torch.matmul(a, a.transpose(-1, -2)) + 1e-05 # make symmetric positive-definite
        npu_input = torch.matmul(a, a.transpose(-1, -2)) + 1e-05
        npu_input = npu_input.to("npu")
        if format != -1:
            npu_input = npu_input.npu_format_cast(format)
        return cpu_input, npu_input

    def cpu_op_exec(self, input1): 
        output = torch.cholesky(input1)
        output = output.numpy() 
        return output 

    def cpu_op_exec_fp16(self, input1):
        output = torch.cholesky(input1)
        output = output.numpy()
        output = output.astype(np.float16)
        return output
    
    def npu_op_exec(self, input1): 
        output = torch.cholesky(input1)
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
    
    def npu_op_exec_fp16(self, input1): 
        output = torch.cholesky(input1)
        output = output.to("cpu") 
        output = output.numpy() 
        output = output.astype(np.float16)
        return output 

    def test_cholesky_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (1, 1)]],
            [[np.float32, -1, (2, 2)]],
            [[np.float32, -1, (4, 4)]],
            [[np.float32, -1, (8, 8)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = self.create_2d_tensor(item[0], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cholesky_float16_shape_format(self, device):
        shape_format = [
            [[np.float16, -1, (4, 2, 4, 4)]],
            [[np.float16, -1, (2, 3, 4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = self.create_nd_tensor(item[0], 1, 2)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec_fp16(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cholesky_float16_2_shape_format(self, device):
        shape_format = [
            [[np.float16, -1, (2, 4, 4)]],
            [[np.float16, -1, (3, 8, 8)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = self.create_nd_tensor(item[0], 1, 2)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec_fp16(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestCholesky, globals(), except_for='cpu')
if __name__ == '__main__': 
    run_tests()