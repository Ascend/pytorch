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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import graph_mode

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
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        res_forward = res_forward.detach().numpy()
        return res_forward, input1.grad, weight.grad
        
    def npu_op_exec(self, input1, weight,  bias1, stride=1, padding=0, dilation=2, groups=1):
        weight1 = weight
        
        input1.requires_grad = True
        #input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        #weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1 = bias1.to("npu")
        bias1.requires_grad = True        
        
        res_forward =torch.nn.functional.conv2d(input1, weight,  bias1, stride, padding, dilation, groups)
        
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        grads = grads.to("npu")
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward, input1.grad.to("cpu"), weight.grad.to("cpu")

    @graph_mode
    def test_slow_conv_dilated2d_backward_shape_format(self, device):

        weight_grad = []
        input_grad = []
                
        shape_format = [
                [np.float32, 0, (64, 1, 16, 14)],
                [np.float32, 3, (256, 1, 8, 8)],
                [np.float32, 4, (32, 1, 8, 8)],
                [np.float32, 0, (10, 1, 16, 16)]
        ]
                     
        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear() 
            cpu_input1, npu_input1 = create_common_tensor(item, -2, 2)
            
            cpu_weight, npu_weight = create_common_tensor([np.float32, 0, (3, 1, 2, 2)], -2, 2)
            
            cpu_bias, npu_bias = create_common_tensor([np.float32, 0, (3)], 1, 100)
            
            cpu_output, cpu_input_grad, cpu_weight_grad =self.cpu_op_exec(cpu_input1, cpu_weight, bias1=cpu_bias)
            npu_output, npu_input_grad, npu_weight_grad = self.npu_op_exec(npu_input1, npu_weight, bias1=npu_bias)
            
            self.assertRtolEqual(cpu_output, npu_output, 0.001)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad, 0.01)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 0.01)

            
instantiate_device_type_tests(TestSlowConvDilated2dBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()