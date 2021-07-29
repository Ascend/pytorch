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
import torch.onnx.symbolic_registry as sym_registry


import torch.utils.cpp_extension
import torch.nn as nn
import numpy as np

def do_export(model, inputs, *args, **kwargs):
    out = torch.onnx._export(model, inputs, "custom_AdaptiveMaxPool2d_op_demo.onnx", verbose=True, *args, **kwargs)

###########################################################


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.adaptiveMaxPool2d = nn.AdaptiveMaxPool2d((5,7),return_indices=False)

    def forward(self, input):

        rtn = self.adaptiveMaxPool2d(input)
        return rtn


############################################################

def i_adaptive_max_pool2d(g, self, a):
    
    rtn = g.op('torch::adaptive_max_pool2d', self, a)

    return rtn,rtn

import torch.onnx.symbolic_registry as sym_registry

sym_registry.register_op('adaptive_max_pool2d', i_adaptive_max_pool2d, '', 11)

print('=======================================================================================')
print(sym_registry.is_registered_op('adaptive_max_pool2d', '',11))
print('=======================================================================================')
############################################################
# device = 'cpu'
#print('{}-{}-{}-{}-{}-{}-{}'.format(a,b,c,d,e,f,g))
model = CustomModel()
input_x_shape = [1, 64, 8, 9]
input = torch.randn(input_x_shape)
output = model(input)
print("output",output)
do_export(model, input, opset_version=11)





