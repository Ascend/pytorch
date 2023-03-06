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

import functools
import os
import random
import sys
from typing import Callable, Optional

import numpy as np
import torch

import torch_npu
from . import wrap_tensor, wrap_torch, wrap_functional
from .module import HOOKModule


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


def default_schedule(_: int):
    return True


def register_hook(model, hook, **kwargs):
    assert hasattr(model, "named_modules"), "Please register hooks to nn.Module."

    seed = kwargs.get('seed', 1234)
    seed_all(seed)

    torch_npu._C._clear_overflow_npu()

    sample = kwargs.get('sample', True)
    pid = os.getpid()
    path = kwargs.get('path', './')

    sched = kwargs.get('schedule', default_schedule)
    step_schedule.set_schedule(sched)

    hook = functools.partial(hook, sample=sample, pid=pid, path=path, capacity=kwargs.get('capacity'))
    initialize_hook(hook)

    for _, module in model.named_modules():
        if not isinstance(module, HOOKModule):
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


class StepSchedule(object):
    def __init__(self):
        self.step_num = 0
        self.schedule = default_schedule

    def set_schedule(self, sched: Optional[Callable[[int], bool]]):
        self.schedule = sched

    def is_step_enable(self):
        return self.schedule(self.step_num)

    def step(self):
        self.step_num += 1

    def reset_step(self):
        self.step_num = 0


step_schedule = StepSchedule()


def schedule(begin: int = 0, end: int = sys.maxsize, stride: int = 1) -> Callable:
    """
    Returns a callback that can be used as register_hook schedule argument.
    Schedule is used to control if hook is enabled for a step.
    Args:
        begin: first enable step
        end: last enable step
        stride: repeat the cycle enable with the next step
    Returns: schedule callback fn
    """
    if begin < 0 or end < 0 or stride <= 0 or begin > end:
        raise RuntimeError("Illegal Argument")

    def schedule_fn(step: int) -> bool:
        if step > end or step < begin:
            return False
        if 0 == (step - begin) % stride:
            return True
        return False

    return schedule_fn
