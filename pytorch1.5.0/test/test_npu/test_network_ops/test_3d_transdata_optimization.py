# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import torch.nn as nn
import numpy as np

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TypicalConvBnNet(nn.Module):
    def __init__(self, dim_c):
        super().__init__()
        self.conv = nn.Conv3d(dim_c, dim_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(dim_c)
    
    def forward(self, input_tensor):
        conv_out = self.conv(input_tensor)
        bn_out = self.bn(conv_out)
        return bn_out
    
class Test3DTransdataOptimization(TestCase):
    weight_grad = []
    input_grad = []

    def get_input_grad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def get_weight_grad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input_tensor, weight_tensor, dim_c):
        input_tensor.requires_grad_(True)
        input_tensor.register_hook(lambda grad: self.get_input_grad(grad))
        conv_bn = TypicalConvBnNet(dim_c)
        conv_bn.conv.weight.data = weight_tensor
        conv_bn.conv.weight.register_hook(lambda grad: self.get_weight_grad(grad))   
        
        input_tensor_out = conv_bn(input_tensor)    
        input_tensor_out.backward(torch.ones_like(input_tensor_out))
        return input_tensor_out

    def npu_op_exec(self, input_tensor, weight_tensor, dim_c):
        input_tensor.requires_grad_(True)
        input_tensor.register_hook(lambda grad: self.get_input_grad(grad))
        conv_bn = TypicalConvBnNet(dim_c).npu()   
        conv_bn.conv.weight.data = weight_tensor
        conv_bn.conv.weight.register_hook(lambda grad: self.get_weight_grad(grad))   
        
        input_tensor_out = conv_bn(input_tensor)
        input_tensor_out.backward(torch.ones_like(input_tensor_out))        
        return input_tensor_out
    
    def test_3d_transdata_optimization_using_conv_and_bn(self, device):
        dtype_format_shape = [  
            # input, weight                
            [[np.float32, 30, [128, 256, 2, 7, 7]], [np.float32, 30, [256, 256, 3, 3, 3]]],
            [[np.float32, 32, [128, 256, 2, 7, 7]], [np.float32, 33, [256, 256, 3, 3, 3]]]
        ]
        for item in dtype_format_shape:
            self.weight_grad.clear()
            self.input_grad.clear()
            input_cpu, input_npu = create_common_tensor(item[0], 0, 1)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)

            # op_exec(inpu, weight, in_channels/out_channels=dim_c)
            output_cpu = self.cpu_op_exec(input_cpu, weight_cpu, item[0][2][1])
            output_npu = self.npu_op_exec(input_npu, weight_npu, item[0][2][1])
            
            output_cpu = output_cpu.to(output_npu.dtype)
            self.input_grad[0] = self.input_grad[0].to(self.input_grad[1].dtype)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(output_cpu.detach().numpy(), output_npu.cpu().detach().numpy())
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].cpu().numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].cpu().numpy())

instantiate_device_type_tests(Test3DTransdataOptimization, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()