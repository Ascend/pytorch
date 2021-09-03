import torch
import torch.utils.cpp_extension

import numpy as np

def do_export(model, inputs, *args, **kwargs):
    out = torch.onnx._export(model, inputs, "custom_demo.onnx", *args, **kwargs)

###########################################################

def test_custom_add():
    op_source = """
    #include <torch/script.h>

    torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
      return self + other;
    }

    static auto registry =
      torch::RegisterOperators("custom_namespace::custom_add", &custom_add);


    """


    torch.utils.cpp_extension.load_inline(
        name="custom_add",
        cpp_sources=op_source,
        is_python_module=False,
        verbose=True,
    )

test_custom_add()

############################################################

class CustomAddModel(torch.nn.Module):

    def forward(self, a, b):

        return torch.ops.custom_namespace.custom_add(a, b)




############################################################

def symbolic_custom_add(g, self, other):
    
    return g.op('custom_namespace::custom_add', self, other)


from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('custom_namespace::custom_add', symbolic_custom_add, 9)


x = torch.randn(2, 3, 4, requires_grad=False)
y = torch.randn(2, 3, 4, requires_grad=False)
model = CustomAddModel()
do_export(model, (x, y), opset_version=11)



