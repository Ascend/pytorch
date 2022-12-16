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
import functools
import random
import numpy as np

import torch

from . import wrap_tensor, wrap_torch, wrap_functional


def initialize_hook(hook):

    wrap_tensor.wrap_tensor_ops_and_bind(hook)
    for attr_name in dir(wrap_tensor.HOOKTensor):
        if attr_name.startswith("wrap_"):
            setattr(torch.Tensor, attr_name[5:], getattr(wrap_tensor.HOOKTensor, attr_name))

    wrap_torch.wrap_torch_ops_and_bind(hook)
    for attr_name in dir(wrap_torch.HOOKTorchOP):
        if attr_name.startswith("wrap_"):
            setattr(torch, attr_name[5:], getattr(wrap_torch.HOOKTorchOP, attr_name))

    wrap_functional.wrap_functional_ops_and_bind(hook)
    for attr_name in dir(wrap_functional.HOOKFunctionalOP):
        if attr_name.startswith("wrap_"):
            setattr(torch.nn.functional, attr_name[5:], getattr(wrap_functional.HOOKFunctionalOP, attr_name))


def register_hook(model, hook, **kwargs):
    assert hasattr(model, "named_modules"), "Please register hooks to nn.Module."

    sample = kwargs.get('sample', True)
    pid = os.getpid()
    hook = functools.partial(hook, sample=sample, pid=pid)
    initialize_hook(hook)
    for _, module in model.named_modules():
        if not hasattr(module, "named_modules") or len(list(module.named_modules())) > 1:
            continue

        prefix = "Module_" + module.__class__.__name__ + "_"
        if hasattr(module, "prefix_op_name_"):
            prefix = module.prefix_op_name_

        module.register_forward_hook(hook(prefix + "forward"))
        module.register_backward_hook(hook(prefix + "backward"))


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
