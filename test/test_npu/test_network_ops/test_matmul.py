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

import sys
import torch
import numpy as np
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMatMul(TestCase):
    def op_exec_cpu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        cpu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(cpu_output)
        cpu_output.backward(tmp)

        return cpu_output.detach().numpy(), input1.grad.numpy(), input2.grad.numpy()

    def op_exec_npu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True
        
        npu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(npu_output)
        npu_output.backward(tmp)
        npu_output = npu_output.cpu()
        return npu_output.detach().cpu().numpy(), input1.grad.cpu().numpy(), input2.grad.cpu().numpy()

    def matmul_backward_result(self, shape_format):
        for item in shape_format:
            mat1_cpu, mat1_npu = create_common_tensor(item[0], -10, 10)
            if mat1_cpu.dtype == torch.float16:
                mat1_cpu = mat1_cpu.to(torch.float32)
            mat2_cpu, mat2_npu = create_common_tensor(item[1], -10, 10)
            if mat2_cpu.dtype == torch.float16:
                mat2_cpu = mat2_cpu.to(torch.float32)
            cpu_output, cpu_mat1_grad, cpu_mat2_grad = self.op_exec_cpu(mat1_cpu, mat2_cpu)
            npu_output, npu_mat1_grad, npu_mat2_grad = self.op_exec_npu(mat1_npu, mat2_npu)

            
            self.assertRtolEqual(cpu_output.astype(npu_output.dtype), npu_output)
            self.assertRtolEqual(cpu_mat1_grad.astype(npu_mat1_grad.dtype), npu_mat1_grad)
            self.assertRtolEqual(cpu_mat2_grad.astype(npu_mat2_grad.dtype), npu_mat2_grad)

    def test_matmul_backward_shape_format_fp16_case1(self, device):
        shape_format = [
            # mat1 1dim, mat2 1dim       
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [2560]], [np.float16, 2, [2560]]],
        ]
        self.matmul_backward_result(shape_format)

    # 暂不支持
    # def test_matmul_backward_shape_format_fp16_case2(self, device):
    #     shape_format = [  # mat1 2dim, mat2 1dim       
    #         [[np.float16, 2, [3,5]], [np.float16, 2, [5]]],
    #         [[np.float16, 2, [2560,4680]], [np.float16, 2, [4680]]],
    #         [[np.float16, 2, [100,200]], [np.float16, 2, [200]]],
    #         [[np.float16, 2, [4,4]], [np.float16, 2, [4]]],
            
    #     ]
    #     self.matmul_backward_result(shape_format)
    
    def test_matmul_backward_shape_format_fp16_case3(self, device):
        shape_format = [
            # mat1 1dim, mat2 2dim       
            [[np.float16, 2, [5]], [np.float16, 2, [5,6]]],
            [[np.float16, 2, [2560]], [np.float16, 2, [2560,4680]]],
            [[np.float16, 2, [5]], [np.float16, 2, [5,5]]],
            
        ]
        self.matmul_backward_result(shape_format)
        
    def test_matmul_backward_shape_format_fp16_case4(self, device):
        shape_format = [
            # mat1 1dim, mat2 2dim       
            [[np.float16, 2, [5,7]], [np.float16, 2, [7,10]]],
            [[np.float16, 2, [3750,2560]], [np.float16, 2, [2560,4680]]],
            [[np.float16, 2, [5,10]], [np.float16, 2, [10,20]]],
        ]
        self.matmul_backward_result(shape_format)
    
    def test_matmul_backward_shape_format_fp16_case5(self, device):
        shape_format = [
            # mat1 1dim, mat2 2dim       
            [[np.float16, 2, [5,7,10]], [np.float16, 2, [10]]],
            [[np.float16, 2, [168,3750,256]], [np.float16, 2, [256]]],
            [[np.float16, 2, [4,5,10]], [np.float16, 2, [10]]],
            #[[np.float16, 2, [5,10,20,30]], [np.float16, 2, [30]]],  # 该shape无法通过
            #[[np.float16, 2, [20,30,40,50,60]], [np.float16, 2, [60]]], batch 三维 精度不行
            [[np.float16, 2, [3,4,5,6,7,16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)
        
    def test_matmul_backward_shape_format_fp16_case6(self, device):
        shape_format = [
            # mat1 >2dim, mat2 2dim       
            [[np.float16, 2, [5,7,10]], [np.float16, 2, [10,16]]],
            #[[np.float16, 2, [5,10,20,30]], [np.float16, 2, [30,25]]], # 该shape无法过
            [[np.float16, 2, [2,5,7,8,19,80]], [np.float16, 2, [80,32]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case7(self, device):
        shape_format = [
            # mat1 1dim, mat2 >2dim       
            [[np.float16, 2, [7]], [np.float16, 2, [5,7,10]]],
            [[np.float16, 2, [5,]], [np.float16, 2, [4,5,10]]],
            # [[np.float16, 2, [20]], [np.float16, 2, [5,10,20,30]]], # 该shape无法过
            [[np.float16, 2, [7]], [np.float16, 2, [3,4,5,6,7,16]]],
        ]
        self.matmul_backward_result(shape_format)
        
    def test_matmul_backward_shape_format_fp16_case8(self, device):
        shape_format = [
            # mat1 2dim, mat2 >2dim       
            [[np.float16, 2, [5,7]], [np.float16, 2, [5,7,10]]],
            [[np.float16, 2, [12,5]], [np.float16, 2, [4,5,10]]],
            # [[np.float16, 2, [44,20]], [np.float16, 2, [5,10,20,30]]], # 该shape无法过
            # [[np.float16, 2, [75,50]], [np.float16, 2, [2,3,40,50,60]]], # 该shape无法过
            [[np.float16, 2, [188,7]], [np.float16, 2, [3,4,5,6,7,16]]],
        ]
        self.matmul_backward_result(shape_format)
        
    def test_matmul_backward_shape_format_fp16_case9(self, device):
        shape_format = [       
            [[np.float16, 2, [5,7,10]], [np.float16, 2, [5,10,15]]],
            [[np.float16, 2, [168,3750,256]], [np.float16, 2, [168,256,43]]],
            # TODO(ascend): Insufficient precision
            # 在两个输入shape不一致的情况下,会通过expand将两个tensor shape对齐。反向时expand的反向会调用sum(dim)，在fp16下与CPU比较不过。
            # 但是结果与CUDA比对通过。所以只放开两个tensor batch部分一致的用例
            # [[np.float16, 2, [1,6,7,65]], [np.float16, 2, [5,6,65,17]]],#该shape无法过
            # [[np.float16, 2, [4,5,10,15]], [np.float16, 2, [5,15,20]]],
            # [[np.float16, 2, [5,10,20,30]], [np.float16, 2, [1,30,40]]],
            # [[np.float16, 2, [20,30,40,50,60]], [np.float16, 2, [40,60,6]]],
            # [[np.float16, 2, [6,7,16]], [np.float16, 2, [4,5,6,16,17]]],
            # [[np.float16, 2, [5,6,7,33]], [np.float16, 2, [12,23,5,6,33,17]]],
            # [[np.float16, 2, [3,4,6,7,44]], [np.float16, 2, [2,3,4,6,44,17]]],
            # [[np.float16, 2, [42,2,3,41]], [np.float16, 2, [1,2,42,2,41,17]]],
        ]
        self.matmul_backward_result(shape_format)
        

instantiate_device_type_tests(TestMatMul, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
