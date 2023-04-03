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

import os
import yaml

import torch

from .module import HOOKModule

cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, "support_wrap_ops.yaml")
with open(yaml_path, 'r') as f:
    WrapFunctionalOps = yaml.safe_load(f).get('functional')


def get_functional_ops():
    global WrapFunctionalOps
    _all_functional_ops = dir(torch.nn.functional)
    assert set(WrapFunctionalOps) <= set(_all_functional_ops)
    return WrapFunctionalOps


class HOOKFunctionalOP(object):
    pass


class FunctionalOPTemplate(HOOKModule):

    def __init__(self, op_name, hook):
        self.op_name_ = op_name
        self.prefix_op_name_ = "Functional_" + str(op_name) + "_"
        super().__init__(hook)

    def forward(self, *args, **kwargs):
        return getattr(torch.nn.functional, str(self.op_name_))(*args, **kwargs)


def wrap_functional_op(op_name, hook):

    def functional_op_template(*args, **kwargs):
        return FunctionalOPTemplate(op_name, hook)(*args, **kwargs)

    return functional_op_template


def wrap_functional_ops_and_bind(hook):
    _functional_ops = get_functional_ops()
    for op_name in _functional_ops:
        setattr(HOOKFunctionalOP, "wrap_" + op_name, wrap_functional_op(op_name, hook))
