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

from typing import List, Set, Any, Dict, Union, Optional
from pathlib import Path
import warnings
import os

import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
from torch.fx import Node, map_arg
from torch.fx.graph import _type_repr, _format_target, _format_args, _get_qualified_name, magic_methods

import torch_npu


def python_code(self, root_module: str) -> str:
    """
    Turn this ``Graph`` into valid Python code.
    Args:
        root_module (str): The name of the root module on which to look-up
            qualified name targets. This is usually 'self'.
    Returns:
        The string source code generated from this ``Graph``.
    """
    free_vars: List[str] = []
    modules_used : Set[str] = set()
    body: List[str] = []

    # Wrap string in list to pass by reference
    maybe_return_annotation : List[str] = ['']

    def register_modules_used(qualified_name : str):
        if '.' in qualified_name:
            module_name = qualified_name.split('.', maxsplit=1)[0]
            modules_used.add(module_name)

    def type_repr(o : Any):
        typename = _type_repr(o)
        if all(x.isidentifier() for x in typename.split('.')):
            register_modules_used(typename)
        else:
            # this is a constructor type, e.g. typing.List[torch.Tensor]
            modules_used.add(o.__module__)
            for sub_type in o.__args__:
                # make sure we have torch.Tensor
                type_repr(sub_type)
        return typename


    # Run through reverse nodes and record the first instance of a use
    # of a given node. This represents the *last* use of the node in the
    # execution order of the program, which we will use to free unused
    # values
    node_to_last_use : Dict[Node, Node] = {}
    user_to_last_uses : Dict[Node, List[Node]] = {}

    def register_last_uses(n : Node, user : Node):
        if n not in node_to_last_use:
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)

    for node in reversed(self.nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    def delete_unused_values(user : Node):
        """
        Delete values after their last use. This ensures that values that are
        not used in the remainder of the code are freed and the memory usage
        of the code is optimal.
        """
        if user.op == 'placeholder':
            return
        if user.op == 'output':
            body.append('\n')
            return
        nodes_to_delete = user_to_last_uses.get(user, [])
        if len(nodes_to_delete):
            to_delete_str = ' = '.join([n.name for n in nodes_to_delete] + ['None'])
            body.append(f';  {to_delete_str}\n')
        else:
            body.append('\n')

    def emit_node(node : Node):
        if node.op == 'placeholder':
            assert isinstance(node.target, str)
            maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
            maybe_default_arg = '' if not node.args else f' = {repr(node.args[0])}'
            free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
            raw_name = node.target.replace('*', '')
            if raw_name != node.name:
                body.append(f'{node.name} = {raw_name}\n')
            return
        elif node.op == 'call_method':
            assert isinstance(node.target, str)
            body.append(
                f'{node.name} = {_format_target(repr(node.args[0]), node.target)}'
                f'({_format_args(node.args[1:], node.kwargs)})')
            return
        elif node.op == 'call_function':
            assert callable(node.target)
            # pretty print operators
            if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                assert isinstance(node.args, tuple)
                body.append(f'{node.name} = \
                    {magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                return
            qualified_name = _get_qualified_name(node.target)
            register_modules_used(qualified_name)
            if qualified_name == 'getattr' and \
                isinstance(node.args, tuple) and \
                isinstance(node.args[1], str) and \
                node.args[1].isidentifier():
                # pretty print attribute access
                body.append(f'{node.name} = {_format_target(repr(node.args[0]), node.args[1])}')
                return
            body.append(f'{node.name} = {qualified_name}({_format_args(node.args, node.kwargs)})')
            return
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            body.append(f'{node.name} = \
                {_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
            return
        elif node.op == 'get_attr':
            assert isinstance(node.target, str)
            body.append(f'{node.name} = {_format_target(root_module, node.target)}')
            return
        elif node.op == 'output':
            if node.type is not None:
                maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
            body.append(f'return {repr(node.args[0])}')
            return
        raise NotImplementedError(f'node: {node.op} {node.target}')

    for node in self.nodes:
        # NOTE: emit_node does not emit a string with newline. It depends
        # on delete_unused_values to append one
        emit_node(node)
        delete_unused_values(node)

    # repr() for inf and nan floating point values aren't parseable by
    # python as literals. Explicitly import the names from the ``math`` module.
    modules_used.add('torch_npu')
    import_strs = [f'import {name}' for name in sorted(modules_used)]
    import_block = '\n'.join(import_strs)

    if len(body) == 0:
        # If the Graph has no non-placeholder nodes, no lines for the body
        # have been emitted. To continue to have valid Python code, emit a
        # single pass statement
        body.append('pass\n')

    code = ''.join(body)
    code = '\n'.join('    ' + line for line in code.split('\n'))
    fn_code = f"""\
{import_block}
def forward(self, {', '.join(free_vars)}){maybe_return_annotation[0]}:
{code}"""

    return fn_code


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
    torch.fx.Graph.python_code = python_code
    torch.fx.GraphModule.to_folder = to_folder
