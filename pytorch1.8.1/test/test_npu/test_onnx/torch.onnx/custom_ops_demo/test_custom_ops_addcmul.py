import torch.utils.cpp_extension
import torch.nn as nn
import numpy as np

def do_export(model, inputs, *args, **kwargs):
    out = torch.onnx._export(model, inputs, "./onnx/custom_op_addcmul.onnx", verbose=True, *args, **kwargs)

###########################################################


class CustomAddModel(torch.nn.Module):

    def forward(self, tensor, tensor1, tensor2, out=None):
        rtn = torch.addcmul(tensor, tensor1, tensor2, value=1.0)
        return rtn


############################################################

def addcmul(g, self, tensor1, tensor2, value=1.0, out=None):
    return g.op('torch::addcmul', self, tensor1, tensor2, value)  

import torch.onnx.symbolic_registry as sym_registry

sym_registry.register_op('addcmul', addcmul, '', 11)

print('=======================================================================================')
print(sym_registry.is_registered_op('addcmul', '',11))
print('=======================================================================================')
############################################################
# device = 'cpu'
#print('{}-{}-{}-{}-{}-{}-{}'.format(a,b,c,d,e,f,g))
model = CustomAddModel()
t = torch.randn(1, 3)
t1 = torch.randn(3, 1)
t2 = torch.randn(1, 3)
#output = model(t)
#print("output",output)

do_export(model, (t,t1,t2), opset_version=11)
