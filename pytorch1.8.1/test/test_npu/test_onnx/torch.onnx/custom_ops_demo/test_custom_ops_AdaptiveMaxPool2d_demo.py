
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





