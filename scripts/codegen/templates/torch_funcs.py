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

import dis
from functools import wraps

import torch

import torch_npu
from torch_npu.utils.device_guard import torch_device_guard


def wrap_torch_warning_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.warned:
            print(f"Warning: torch.{func.__name__} is deprecated and will be removed in future version. "
                  f"Use torch_npu.{func.__name__} instead.")
            wrapper.warned = True
            return func(*args, **kwargs)
    wrapper.warned = False
    return wrapper


@torch_device_guard
def _tensor(*args, **kwargs):
    return torch_npu.tensor(*args, **kwargs)


@torch_device_guard
def _full(*args, **kwargs):
    return torch_npu.full(*args, **kwargs)


@torch_device_guard
def _randint(*args, **kwargs):
    return torch_npu.randint(*args, **kwargs)


@torch_device_guard
def _range(*args, **kwargs):
    return torch_npu.range(*args, **kwargs)


@torch_device_guard
def _arange(*args, **kwargs):
    return torch_npu.arange(*args, **kwargs)


@wrap_torch_warning_func
@torch_device_guard
def _empty_with_format(*args, **kwargs):
    return torch_npu.empty_with_format(*args, **kwargs)


@torch_device_guard
def _npu_dropout_gen_mask(*args, **kwargs):
    return torch_npu.npu_dropout_gen_mask(*args, **kwargs)


@torch_device_guard
def _new_device(*args, **kwargs):
    return torch_npu._C.device(*args, **kwargs)


def jit_script(obj, optimize=None, _frames_up=0, _rcb=None):
    # (Ascend) Disable extension of torch.jit.script
    return obj


def _as_tensor(*args, **kwargs):
    if isinstance(args[0], torch.Tensor):
        dst_device = args[0].device
    else:
        dst_device = "cpu"

    if kwargs and "device" in kwargs:
        dst_device = kwargs.pop("device")

    return torch._C._VariableFunctions.as_tensor(*args, **kwargs).to(dst_device)


def _eval_no_call(stmt, glob, loc):
    """Evaluate statement as long as it does not contain any method/function calls"""
    bytecode = compile(stmt, "", mode="eval")
    for insn in dis.get_instructions(bytecode):
        if "CALL" in insn.opname:
            raise RuntimeError(f"Type annotation should not contain calls, but '{stmt}' does")
    return eval(bytecode, glob, loc)


def _parse_type_line(type_line, rcb, loc):
    """Parses a type annotation specified as a comment.
    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
    arg_ann_str, ret_ann_str = torch.jit.annotations.split_type_line(type_line)

    try:
        arg_ann = _eval_no_call(arg_ann_str, {}, torch.jit.annotations.EvalEnv(rcb))
    except (NameError, SyntaxError) as e:
        raise RuntimeError("Failed to parse the argument list of a type annotation") from e

    if not isinstance(arg_ann, tuple):
        arg_ann = (arg_ann,)

    try:
        ret_ann = _eval_no_call(ret_ann_str, {}, torch.jit.annotations.EvalEnv(rcb))
    except (NameError, SyntaxError) as e:
        raise RuntimeError("Failed to parse the return type of a type annotation") from e

    arg_types = [torch.jit.annotations.ann_to_type(ann, loc) for ann in arg_ann]
    return arg_types, torch.jit.annotations.ann_to_type(ret_ann, loc)


${device_methods_def_py_dispatch}

def add_torch_funcs():
    torch.tensor = _tensor
    torch.full = _full
    torch.randint = _randint
    torch.range = _range
    torch.arange = _arange
    torch.empty_with_format = _empty_with_format
    torch.npu_dropout_gen_mask = _npu_dropout_gen_mask
    torch.jit.script = jit_script
    torch.as_tensor = _as_tensor
    torch.new_device = _new_device
    torch.jit.annotations.parse_type_line = _parse_type_line

${device_methods_def_py}
