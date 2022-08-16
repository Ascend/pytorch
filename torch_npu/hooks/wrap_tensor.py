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


WrapTensorOps = [
    '__add__', '__div__', '__idiv__', '__isub__', '__mul__', '__sub__', 'abs', 'abs_', 'acos', 
    'acos_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_',
    'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'baddbmm', 'baddbmm_', 'bernoulli',
    'bernoulli_', 'bitwise_and', 'bitwise_and_', 'bitwise_not', 'bitwise_not_', 'bitwise_or',
    'bitwise_or_', 'bitwise_xor', 'bitwise_xor_', 'bmm', 'ceil', 'ceil_', 'clamp', 'clamp_',
    'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'cos', 'cos_', 'cosh', 'cosh_', 'div',
    'div_', 'dot', 'softmax'
]


def get_tensor_ops():
    global WrapTensorOps
    _tensor_ops = dir(torch._C._TensorBase)
    assert set(WrapTensorOps) <= set(_tensor_ops)
    return WrapTensorOps


class HOOKTensor(object):
    pass


class TensorOPTemplate(HOOKModule):
    
    def __init__(self, op_name):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Tensor_" + str(op_name) + "_"
        super().__init__()

    def forward(self, *args, **kwargs):
        return getattr(torch._C._TensorBase, str(self.op_name_))(*args, **kwargs)


def wrap_tensor_op(op_name):
    def tensor_op_template(*args, **kwargs):
        return TensorOPTemplate(op_name)(*args, **kwargs)
    return tensor_op_template


def wrap_tensor_ops_and_bind():
    _tensor_ops = get_tensor_ops()
    for op_name in _tensor_ops:
        setattr(HOOKTensor, "wrap_" + str(op_name), wrap_tensor_op(op_name))
