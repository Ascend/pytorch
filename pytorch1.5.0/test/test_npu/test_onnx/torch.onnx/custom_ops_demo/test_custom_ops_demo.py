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



