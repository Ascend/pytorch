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


class HOOKTensor(object):
    
    def wrap___add__(self, other):
        return Add()(self, other)
    
    def wrap___add__(self, rother):
        return Add()(rother, self)

    def wrap___sub__(self, other):
        return Sub()(self, other)
    
    def wrap___rsub__(self, rother):
        return Sub()(rother, self)
    
    def wrap___truediv__(self, other):
        return Div()(self, other)
    
    def wrap___rtruediv__(self, rother):
        return Div()(rother, self)
    
    def wrap___mul__(self, other):
        return Mul()(self, other)
    
    def wrap___rmul__(self, rother):
        return Mul()(rother, self)
    
    def add(self, *args, **kwargs):
        return Add()(self, *args, **kwargs)


class Add(HOOKModule):
    
    def forward(self, *args, **kwargs):
        return torch._C._VariableFunctions.add(*args, **kwargs)


class Sub(HOOKModule):
    
    def forward(self, *args, **kwargs):
        return torch._C._VariableFunctions.sub(*args, **kwargs)


class Div(HOOKModule):
    
    def forward(self, *args, **kwargs):
        return torch._C._VariableFunctions.div(*args, **kwargs)


class Mul(HOOKModule):
    
    def forward(self, *args, **kwargs):
        return torch._C._VariableFunctions.mul(*args, **kwargs)
