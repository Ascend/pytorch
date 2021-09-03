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

import copy
import torch
import numpy as np
from torch.testing._internal.common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

def getWeightGrad(self, grad):
    self.weight_grad.append(grad.to("cpu"))

def getInputGrad(self, grad):
    self.input_grad.append(grad.to("cpu"))

class TestSlowConvDilated2dBackward(TestCase): 
    weight_grad = []
    input_grad = []
    
    def cpu_op_exec(self, input1, weight,  bias1, stride=1, padding=0, dilation=2, groups=1):
        weight1 = weight
        
        input1.requires_grad = True
        #input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        #weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1.requires_grad = True
        
        res_forward = torch.nn.functional.conv2d(input1, weight, bias1, stride, padding, dilation, groups)
        print("===cpu_res_forward===")
        print(res_forward)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        print("===cpu_bias===")
        print(bias1)
        print("===cpu_bias_grad===")
        print(bias1.grad)
        res_forward = res_forward.detach().numpy()
        return res_forward
        
    def npu_op_exec(self, input1, weight,  bias1, stride=1, padding=0, dilation=2, groups=1):
        weight1 = weight
        
        input1.requires_grad = True
        #input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        #weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1 = bias1.to("npu")
        bias1.requires_grad = True        
        
        res_forward =torch.nn.functional.conv2d(input1, weight,  bias1, stride, padding, dilation, groups)
        
        print("===npu_res_forward===")
        print(res_forward)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        grads = grads.to("npu")
        print("===npu_bias===")
        print(bias1)
        print("===npu_bias_grad===")
        print(bias1.grad)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward

    def test_slow_conv_dilated2d_backward_shape_format(self, device):

        weight_grad = []
        input_grad = []
                
        shape_format = [
                [np.float32, 0, (64, 1, 16, 14)],
                [np.float32, 1, (64, 10, 16, 14)],
                [np.float32, 3, (256, 1, 8, 8)],
                [np.float32, 4, (32, 1, 2, 2)],
                [np.float32, 29, (10, 1, 16, 16)]
        ]
                     
        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear() 
            cpu_input1, npu_input1 = create_common_tensor(item, -2, 2)
            
            cpu_weight, npu_weight = create_common_tensor([np.float32, 0, (3, 1, 2, 2)], -2, 2)
            
            cpu_bias, npu_bias = create_common_tensor([np.float32, 0, (3)], 1, 100)
            
            npu_weight =npu_weight.to("cpu") 
            
            cpu_output =self.cpu_op_exec(cpu_input1, cpu_weight, bias1=cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_weight, bias1=npu_bias)
            print("===input_grad===")
            print(self.input_grad)
            print("===weight_grad===")
            print(self.weight_grad)
            
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(self.input_grad[0], self.input_grad[1])
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1])

            
instantiate_device_type_tests(TestSlowConvDilated2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()