
import torch
import torch.onnx.symbolic_registry as sym_registry

import torch.utils.cpp_extension
import torch.nn as nn
import torch.nn.modules as Module
from torch.autograd import Function
import numpy as np

from torch.nn.parameter import Parameter
import math
from torch.nn  import init

ONNX_FILE_NAME = "./custom_python_module_demo.onnx"
def do_export(model, inputs, *args, **kwargs):
    out = torch.onnx._export(model, inputs, ONNX_FILE_NAME, verbose=True,keep_initializers_as_inputs=True, *args, **kwargs)


############################################################
#Function 
############################################################
class CustomClassOp_Add_F(Function):
    @staticmethod
    def forward(ctx, input1,input2):
        rtn = torch.add(input1,input2)
        return torch.add(input1,rtn)

    @staticmethod
    def symbolic(g,input1,input2):
        rtn = g.op("Custom::CustomClassOp_Add", input1, input2,test_attr1_i=1,test_attr2_f=1.0)
        rtn = g.op("ATen::CustomClassOp_Add", input1, rtn)
        rtn = g.op("C10::CustomClassOp_Add", rtn, input2)
        #erro doman: rtn = g.op("onnx::CustomClassOp_Add", input1, input2)

        return rtn


############################################################
#custom class op
############################################################
class CustomClassOp_Add(torch.nn.Module):
    def __init__(self):
        super(CustomClassOp_Add, self).__init__()
        self.add = CustomClassOp_Add_F.apply

        #graph(%0 : Float(1, 8, 10, 1024),
        #      %1 : Float(8, 10, 1024))
        self.weight = Parameter(torch.Tensor(8,10,1024))

        #%1 : Float(8, 10, 1024) = onnx::Constant[value=<Tensor>]()
        self.weight1 = torch.Tensor(8,10,1024)

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
    def forward(self, input):
        rtn = torch.add(self.weight1, self.weight)

        rtn = self.add(self.weight, rtn)
        rtn1 = self.add(self.weight, self.weight1)
        rtn1 = self.add(self.weight1,rtn1) 
        rtn = self.add(rtn,rtn1)

        return rtn
############################################################
#custom Model
############################################################

class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.add = CustomClassOp_Add()
        self.weight = Parameter(torch.Tensor(8,10,1024))

    def forward(self, input):
        rtn = self.add(input)
        rtn = torch.add(self.weight, rtn)
        return rtn


############################################################
def test_class_export():
    model = CustomModel()
    model.eval()
    input_x_shape = [1, 8, 10, 1024]
    input = torch.randn(input_x_shape)
    output = model(input)
    do_export(model, input, opset_version=11)


import onnx
def check_onnxfile():
    print("test onnx file:",ONNX_FILE_NAME)
    # Load the ONNX model
    model = onnx.load(ONNX_FILE_NAME)
    #print(model)
    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))
    
test_class_export()

check_onnxfile()
