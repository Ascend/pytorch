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

from .module import HOOKModule

_torch_include_ops = ['add', 'sub', 'relu', 'batch_norm', 'conv2d']

def get_torch_ops():
    global _torch_include_ops
    _torch_ops = dir(torch._C._VariableFunctionsClass)
    assert set(_torch_include_ops) <= set(_torch_ops)
    return _torch_include_ops


class HOOKTorchOP(object):
    pass


class TorchOPTemplate(HOOKModule):
    
    def __init__(self, op_name):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Torch_" + str(op_name) + "_"
        super().__init__()

    def forward(self, *args, **kwargs):
        return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)


def wrap_torch_op(op_name):
    def torch_op_template(*args, **kwargs):
        return TorchOPTemplate(op_name)(*args, **kwargs)
    return torch_op_template


def add_torch_ops_hook(torch_dirs):
    torch_op_dirs_dict = {}
    for op_name in torch_dirs:
        torch_op_dirs_dict[op_name] = wrap_torch_op(op_name)
    return torch_op_dirs_dict


def wrap_torch_ops_and_bind():
    _torch_op_dirs = get_torch_ops()
    torch_op_dirs_dict = add_torch_ops_hook(_torch_op_dirs)
    for key, value in torch_op_dirs_dict.items():
        setattr(HOOKTorchOP, "wrap_" + str(key), value)
