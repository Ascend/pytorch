# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import copy
import contextlib
import functools
import sympy

import torch
import torch._functorch.config as functor_config
from torch._functorch.aot_autograd import aot_module_simplified
from torch.fx.node import Argument, Target
from torch.fx import Interpreter
from typing import Any, Dict, List, Tuple, Union
from torch.utils._mode_utils import no_dispatch
from torch._tensor import Tensor
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from torch_npu.npu.replay_graph import enable_replay_graph_mode
from torch_npu.npu.replay_graph import generate_replay_graph
from torch_npu.npu.replay_graph import ReplayGraph


def suppress_guards_get_int(v: torch.SymInt, shape_env: ShapeEnv = None) -> int:
    used_env = shape_env if shape_env else copy.deepcopy(v.node.shape_env)
    with used_env.suppress_guards():
        return int(used_env.evaluate_expr(v.node.expr))


def get_real_shape_size(fake: torch.Size, shape_env: ShapeEnv = None) -> List[int]:
    return [suppress_guards_get_int(dim, shape_env) if isinstance(dim, torch.SymInt) else dim for dim in fake]


def real_tensor_like(fake: torch.Tensor, device=None, shape_env: ShapeEnv = None) -> torch.Tensor:
    with no_dispatch():
        return torch.empty(get_real_shape_size(fake.size(), shape_env), dtype=fake.dtype, device=device)


def is_zero_tensor(t: torch.Tensor):
    if any([isinstance(v, torch.SymInt) for v in t.size()]):
        return False
    return not all(t.size())


@contextlib.contextmanager
def enable_npu_graph_tensor_repr():
    """
    :return: Context for npu graph tensor pretty print
    """

    def _(self: torch.Tensor):
        return f'graph node tensor({self.size()}, {self.dtype}, {self.device})'

    prior = Tensor.__repr__
    try:
        Tensor.__repr__ = _
        yield
    finally:
        Tensor.__repr__ = prior


class NpuGraphMeta:
    def __init__(self, input_shape_ranges, input_sym_shapes, output_sym_shapes):
        self.input_shape_ranges = input_shape_ranges
        self.input_sym_shapes = input_sym_shapes
        self.output_sym_shapes = output_sym_shapes

    def infer_output_shapes(self, input_tensors):
        output_shapes = []
        shape_env = ShapeEnv()
        for sym_dims, real_sizes in zip(self.input_sym_shapes, [t.size() for t in input_tensors]):
            for sym_dim, real_size in zip(sym_dims, real_sizes):
                if isinstance(sym_dim, torch.SymInt):
                    shape_env.var_to_val[sym_dim.node.expr] = sympy.Integer(real_size)
        with shape_env.suppress_guards():
            for idx, sym_shape in enumerate(self.output_sym_shapes):
                output_shapes.append(get_real_shape_size(sym_shape, shape_env))
        return output_shapes


class StaticNpuGraphMeta:
    def __init__(self, input_shape_ranges):
        self.input_shape_ranges = input_shape_ranges


class NpuTracingMetaInterpreter(Interpreter):
    """
    Interpreter for collect npu graph meta from fx graph, such as sym of output, input shape ranges, etc.
    TODO: Add doc here
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape_ranges = []
        self.input_sym_shapes = []
        self.output_sym_shapes = []

    def _get_shape_range(self, fake: torch.Tensor) -> List[int]:
        shape_range = []
        for dim in fake.size():
            if not isinstance(dim, torch.SymInt):
                shape_range.append(dim)
            else:
                try:
                    shape_range.append(int(str(dim)))
                except:
                    shape_range.append(-1)
        return shape_range

    def run(self, *args) -> [NpuGraphMeta, StaticNpuGraphMeta]:
        for arg in args:
            self.input_shape_ranges.append(self._get_shape_range(arg))
        if any([-1 in shape_range for shape_range in self.input_shape_ranges]):
            super(NpuTracingMetaInterpreter, self).run(*args)
            return NpuGraphMeta(self.input_shape_ranges, self.input_sym_shapes, self.output_sym_shapes)
        return StaticNpuGraphMeta(self.input_shape_ranges)

    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        v = super().placeholder(target, args=args, kwargs=kwargs)
        self.input_sym_shapes.append(v.size())
        return v

    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        v_tuple = super().output(target, args=args, kwargs=kwargs)
        for v in v_tuple:
            if isinstance(v, torch.Tensor) and not is_zero_tensor(v):
                self.output_sym_shapes.append(v.size())
            else:
                pass  # TODO: maybe log something here
        return v_tuple


class GraphOutput(object):
    def __init__(self, index):
        self.index = index


class OutputBindings(object):
    def __init__(self):
        self.bindings = []

    def append(self, v):
        self.bindings.append(v)

    def bind(self, graph_outputs: List[torch.Tensor]):
        return [i if not isinstance(i, GraphOutput) else graph_outputs[i.index] for i in self.bindings]


class NpuTracingGraphInterpreter(Interpreter):
    """
    Interpreter for trans fx graph to npu graph
    TODO: Add doc here
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_inputs = []
        self.graph_outputs = []
        self.stateful_ops = []
        self.output_bindings = OutputBindings()  # mapping npu graph output to fx graph output
        self._graph_output_index = 0
        super().__init__(*args, **kwargs)

    def next_graph_output(self):
        v = GraphOutput(self._graph_output_index)
        self._graph_output_index += 1
        return v

    def run(self, *args):
        with enable_npu_graph_tensor_repr(), enable_replay_graph_mode():
            super(NpuTracingGraphInterpreter, self).run(*args)
            return generate_replay_graph(None, self.graph_inputs, [], self.graph_outputs), self.output_bindings

    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        v = super().placeholder(target, args=args, kwargs=kwargs)
        if isinstance(v, torch.Tensor):
            # TODO: We should support zero-tensor(tensor with 0 dim size) output
            assert not is_zero_tensor(v), f'Unsupported zero-tensor input {v}'
            self.graph_inputs.append(v)
            print(f'Collect fx graph input {len(self.graph_inputs)}: {str(v)}')
        else:
            print(f'Skip non-tensor input {v}')
        return v

    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        v_tuple = super().output(target, args=args, kwargs=kwargs)
        for v in v_tuple:
            if isinstance(v, torch.Tensor):
                if is_zero_tensor(v):
                    # TODO: We should support zero-tensor(tensor with 0 dim size) output
                    print(f'Bypass zero-tensor fx graph output {v}')
                    self.output_bindings.append(v)
                else:
                    print(f'Collect fx graph output {len(self.graph_outputs)}: {str(v)}')
                    self.graph_outputs.append(v)
                    self.output_bindings.append(self.next_graph_output())
            else:
                print(f'Bypass non-tensor fx graph output {v}')
                self.output_bindings.append(v)
        return v_tuple


def npu_fx_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # TODO: Using logger instead of print
    print(f'Npu fx compiler compiling graph module:')
    gm.graph.print_tabular()

    with no_dispatch():
        real_inputs = [real_tensor_like(example, device=example.device) for example in example_inputs]
        print('Tracing npu graph with tensor:')
        print('\n'.join([f'tensor({i.size()}, {i.dtype}, {i.device})' for i in real_inputs]))

        real_inputs_device = []
        for arg in real_inputs:
            real_inputs_device.append(arg.npu())

        replay_graph, output_bindings = NpuTracingGraphInterpreter(gm).run(*real_inputs_device)

    print('Tracing npu graph meta with tensor:')
    print('\n'.join([f'fake tensor({i.size()}, {i.dtype}, {i.device})' for i in example_inputs]))
    graph_meta = NpuTracingMetaInterpreter(gm).run(*example_inputs)

    print('Compiling npu graph with shape range:')
    print('\n'.join([f'input {i} shape range: {v}' for i, v in enumerate(graph_meta.input_shape_ranges)]))

    # TODO: compiling npu graph here

    def call_graph_replay(*args, replay_graph: ReplayGraph, graph_meta: [NpuGraphMeta, StaticNpuGraphMeta],
                          output_bindings: OutputBindings):

        args_device = []
        for arg in args:
            args_device.append(arg.npu())

        # TODO: replay graph should be public in the future
        return_output = replay_graph._ReplayGraph__replay(args_device, [])

        for i in range(len(args)):
            # TODO: just copy_ when needed
            args[i].copy_(args_device[i])

        return output_bindings.bind(return_output)

    return functools.partial(call_graph_replay, replay_graph=replay_graph, graph_meta=graph_meta,
                             output_bindings=output_bindings)


def npu_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    @functools.wraps(aot_module_simplified)
    def npu_aot_simplified(*args, **kwargs):
        prior = functor_config.use_fake_tensor
        try:
            functor_config.use_fake_tensor = True
            return aot_module_simplified(*args, **kwargs)
        finally:
            functor_config.use_fake_tensor = prior

    return npu_aot_simplified(gm, example_inputs, fw_compiler=npu_fx_compiler)
