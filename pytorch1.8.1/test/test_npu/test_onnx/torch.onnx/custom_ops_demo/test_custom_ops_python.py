
import torch
import torch.onnx.symbolic_registry as sym_registry

import torch.utils.cpp_extension
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def do_export(model, inputs, *args, **kwargs):
    out = torch.onnx._export(model, inputs, "custom_python_op_demo.onnx", verbose=True, *args, **kwargs)

############################################################
#class op
############################################################
class CustomClassOp_Add(Function):
    @staticmethod
    def forward(ctx, input):
        #return torch.Add(input, input)
        return input

    @staticmethod
    def symbolic(g, self):
        #rtn=g.op('CustomAdd', self)
        rtn=g.op('CustomAdd', self,self)
        return rtn


#############################

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        #self.adaptiveMaxPool2d = CustomClassOp_AdaptiveMaxPool2d((5,7),return_indices=False)
        self.add = CustomClassOp_Add.apply

    def forward(self, input):
        rtn = self.add(input)
        return rtn


############################################################
# device = 'cpu'
#print('{}-{}-{}-{}-{}-{}-{}'.format(a,b,c,d,e,f,g))

def test_class_export():
    model = CustomModel()
    model.eval()
    input_x_shape = [1, 64, 8, 9]
    input = torch.randn(input_x_shape)
    output = model(input)
    do_export(model, input, opset_version=11)


test_class_export()
