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

from typing import List, Set, Any, Dict, Union, Optional, NamedTuple
from pathlib import Path
import warnings
import math
import os

import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
from torch.fx import Node, map_arg
from torch.fx.graph import _type_repr, _format_target, _format_args, _get_qualified_name, magic_methods
from torch.fx import _pytree as fx_pytree
import torch.utils._pytree as pytree

import torch_npu


class _CustomBuiltin(NamedTuple):
    """Additional objs that we add to every graph's globals.
    The repr() for some standard library objects is not valid Python code without
    an import. For common objects of this sort, we bundle them in the globals of
    every FX graph.
    """
    # How to import this object from the standard library.
    import_str: str
    # The actual object, produced from that import string.
    obj: Any

_custom_builtins: Dict[str, _CustomBuiltin] = {}

def _register_custom_builtin(name: str, import_str: str, obj: Any):
    _custom_builtins[name] = _CustomBuiltin(import_str, obj)

_register_custom_builtin('inf', 'from math import inf', math.inf)
_register_custom_builtin('nan', 'from math import nan', math.nan)
_register_custom_builtin('NoneType', 'NoneType = type(None)', type(None))
_register_custom_builtin('torch', 'import torch', torch)
_register_custom_builtin('device', 'from torch import device', torch.device)
_register_custom_builtin('fx_pytree', 'import torch.fx._pytree as fx_pytree', fx_pytree)
_register_custom_builtin('pytree', 'import torch.utils._pytree as pytree', pytree)
_register_custom_builtin('torch_npu', 'import torch_npu', torch_npu)

def to_folder(self, folder: Union[str, os.PathLike], module_name : str = "FxModule"):
    """Dumps out module to ``folder`` with ``module_name`` so that it can be
    imported with ``from <folder> import <module_name>``
    Args:
        folder (Union[str, os.PathLike]): The folder to write the code out to
        module_name (str): Top-level name to use for the ``Module`` while
            writing out the code
    """
    folder = Path(folder)
    Path(folder).mkdir(exist_ok=True)
    torch.save(self.state_dict(), folder / 'state_dict.pt')
    tab = " " * 4
    model_str = f"""
import torch
from torch.nn import *
class {module_name}(torch.nn.Module):
    def __init__(self):
        super().__init__()
"""

    def _gen_model_repr(module_name: str, module: torch.nn.Module) -> Optional[str]:
        safe_reprs = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        if type(module) in safe_reprs:
            return f"{module.__repr__()}"
        else:
            return None

    blobified_modules = []
    for module_name, module in self.named_children():
        module_str = _gen_model_repr(module_name, module)
        if module_str is None:
            module_file = folder / f'{module_name}.pt'
            torch.save(module, module_file)
            blobified_modules.append(module_name)
            module_repr = module.__repr__().replace('\r', ' ').replace('\n', ' ')
            module_str = f"torch.load(r'{module_file}') # {module_repr}"
        model_str += f"{tab*2}self.{module_name} = {module_str}\n"

    for buffer_name, buffer in self._buffers.items():
        model_str += f"{tab*2}self.register_buffer('{buffer_name}', torch.empty({list(buffer.shape)}))\n"

    for param_name, param in self._parameters.items():
        model_str += f"{tab*2}self.{param_name} = torch.nn.Parameter(torch.empty({list(buffer.shape)}))\n"

    model_str += f"{tab*2}self.load_state_dict(torch.load(r'{folder}/state_dict.pt'))\n{tab}"
    model_str += f"{_addindent(self.code, 4)}\n"

    module_file = folder / 'module.py'
    module_file.write_text(model_str)

    init_file = folder / '__init__.py'
    init_file.write_text('from .module import *')

    if len(blobified_modules) > 0:
        warnings.warn("Was not able to save the following children modules as reprs -"
                        f"saved as pickled files instead: {blobified_modules}")


def add_fx_methods():
    torch.fx.graph._custom_builtins = _custom_builtins
    torch.fx.GraphModule.to_folder = to_folder
