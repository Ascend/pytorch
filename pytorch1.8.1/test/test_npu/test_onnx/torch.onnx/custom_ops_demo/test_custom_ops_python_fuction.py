
import torch
import torch.onnx.symbolic_registry as sym_registry

import torch.utils.cpp_extension
import torch.nn as nn
from torch.autograd import Function
import numpy as np

from torch.nn.parameter import Parameter
import math
from torch.nn  import init

ONNX_FILE_NAME = "custom_python_function_demo.onnx"

def do_export(model, inputs, *args, **kwargs):
    out = torch.onnx._export(model, inputs, ONNX_FILE_NAME, verbose=True,keep_initializers_as_inputs=True, *args, **kwargs)

############################################################
#class op
############################################################
class CustomClassOp_Add(Function):
    #@staticmethod
    def __init__(self):
        super(CustomClassOp_Add, self).__init__()

        self.weight = Parameter(torch.Tensor(8,10,1024))
        #other method:
        #self.register_parameter('weight', a)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False
        #self.weight.volatile = True

    @staticmethod
    def forward(ctx, input):
        #def forward(self, input):
        #return torch.add(self.weight,input)
        return torch.add(input,input)
        #return input

    @staticmethod
    def symbolic(g,input):
        rtn = g.op("ATen::CustomClassOp_Add", input, input)
        return rtn


#############################

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        #self.adaptiveMaxPool2d = CustomClassOp_AdaptiveMaxPool2d((5,7),return_indices=False)
        self.CustomClassOp_Add = CustomClassOp_Add()
        self.add = self.CustomClassOp_Add.apply

    def forward(self, input):
        #rtn = self.add(self.CustomClassOp_Add, input)
        rtn = self.add(input)
        return rtn


############################################################
# device = 'cpu'
#print('{}-{}-{}-{}-{}-{}-{}'.format(a,b,c,d,e,f,g))

def test_class_export():
    model = CustomModel()
    model.eval()
    input_x_shape = [8, 10, 1024]
    input = torch.randn(input_x_shape)
    #output = model(input)
    do_export(model, input, opset_version=11)

import onnx
def check_onnxfile():
    print("test onnx file:",ONNX_FILE_NAME)
    # Load the ONNX model
    model = onnx.load(ONNX_FILE_NAME)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


test_class_export()
check_onnxfile()



