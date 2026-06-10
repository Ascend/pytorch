# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Licensed under the Apache-2.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/pytorch/pytorch/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import sympy
from functools import reduce
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch._ops
import torch.fx
from sympy.core import Expr, Integer, Symbol
from sympy.core.numbers import Number as SympyNumber
from torch._inductor import ir
from torch._inductor.ir import ExpandView, IndexingConstant, TensorBox
from torch._inductor.virtualized import V

LOWERING_REGISTRY_ATTRS: tuple[str, ...] = (
    "lowerings",
    "_maybe_layout_constraints",
    "fallbacks",
    "needs_realized_inputs",
    "foreach_ops",
    "inplace_foreach_ops",
    "inplaceable_foreach_ops",
)


def get_module_functions(module: Any) -> dict[str, Callable[..., Any]]:
    functions: dict[str, Callable[..., Any]] = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if inspect.getmodule(func) is module:
            functions[name] = func
    return functions

aten = torch.ops.aten
prims = torch.ops.prims


def add_overload(input_list, output_set):
    for fn in input_list:
        output_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                output_set.add(other_fn)


def resolve_op_from_name(op_name: str, logger=None):
    try:
        obj = torch.ops
        for part in op_name.split('.'):
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        if logger is not None:
            logger.warning(f"[npu|inductor|lowering|fallback] invalid identifier name: {op_name}")
        return None


def get_nested_attr(obj, attr_path, default=None):
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default


def fallback_ops_with_meta(lowerings, decompositions, make_fallback, fallback_list=None):
    """
    Fallback all ops that have a Meta implementation but are not yet in lowerings.
    """
    all_ops = torch._C._dispatch_get_all_op_names()

    for op_name in all_ops:
        has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta")
        has_comp = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "CompositeImplicitAutograd")

        if not (has_meta or has_comp):
            continue

        namespace, name_with_overload = op_name.split("::", 1)

        if "." in name_with_overload:
            name, overload = name_with_overload.rsplit(".", 1)
        else:
            name, overload = name_with_overload, "default"

        normalized_path = f"{namespace}.{name}.{overload}"
        op_overload = get_nested_attr(torch.ops, normalized_path)
        if not isinstance(op_overload, torch._ops.OpOverload):
            continue

        if op_overload in lowerings or op_overload in decompositions:
            continue

        make_fallback(op_overload)
        if fallback_list is not None:
            fallback_list.append(op_overload)


def enable_full_lowering_fallback(
    lowerings,
    decompositions,
    make_fallback,
    fallback_list=None,
    excluded_ops=(),
):
    ops_to_fallback = list(filter(
        lambda op: op not in decompositions and
            isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)) and
            op not in excluded_ops,
        lowerings
    ))
    for op in ops_to_fallback:
        make_fallback(op)
        if fallback_list is not None:
            fallback_list.append(op)

    fallback_ops_with_meta(lowerings, decompositions, make_fallback, fallback_list)


class TracedGraph:
    def __init__(self):
        self.graph = torch.fx.Graph()
        self.last_node: Optional[torch.fx.Node] = None
        self.sym_nodes: Dict[str, torch.fx.Node] = {}

    def __str__(self):
        return str(self.graph)

    def get_placeholder_names(self):
        placeholder_names = set()
        for node in self.graph.nodes:
            if node.op == "placeholder" and node.name not in self.sym_nodes:
                placeholder_names.add(node.name)
        return placeholder_names

    __repr__ = __str__


def create_fake_input(size, stride, device, dtype):
    size = [
        V.graph.sizevars.shape_env.create_symintnode(s, hint=None)
        if isinstance(s, Expr) and not isinstance(s, Integer)
        else s
        for s in size
    ]
    stride = [
        V.graph.sizevars.shape_env.create_symintnode(s, hint=None)
        if isinstance(s, Expr) and not isinstance(s, Integer)
        else s
        for s in stride
    ]
    with V.graph.fake_mode:
        fake_input = torch.empty_strided(size, stride, device=device, dtype=dtype)
    return fake_input


def get_reduction_type_to_aten_fn():
    return {
        "sum": aten.sum,
        "prod": aten.prod,
        "xor_sum": prims.xor_sum,
        "any": aten.any,
        "max": aten.amax,
        "min": aten.amin,
        "argmax": aten.argmax,
        "argmin": aten.argmin,
    }


def register_fn_to_aten_fn(registry: Dict[Callable, object], fn, aten_fn=None):
    if fn not in registry:
        registry[fn] = aten_fn
    return fn


def register_to_aten(registry: Dict[Callable, object], aten_fn=None):
    def decorator(fn):
        if fn not in registry:
            registry[fn] = aten_fn
        return fn

    return decorator


@dataclass(frozen=True)
class OperatorMapping:
    """Sympy expr symbol encoding for FX placeholder names."""

    operator_to_string: Dict[str, str]
    encoded_prefix: str
    decoded_strip_len: int

    @property
    def string_to_operator(self) -> Dict[str, str]:
        return {v: k for k, v in self.operator_to_string.items()}


TRITON_OPERATOR_MAPPING = OperatorMapping(
    operator_to_string={
        "+": "a",
        "-": "sub",
        "*": "m",
        "/": "d",
        "(": "l",
        ")": "r",
        ".": "p",
    },
    encoded_prefix="_",
    decoded_strip_len=1,
)

MLIR_OPERATOR_MAPPING = OperatorMapping(
    operator_to_string={
        "+": "xvxa",
        "-": "xvxb",
        "*": "xvxc",
        "/": "xvxd",
        "(": "xvxe",
        ")": "xvxf",
        ".": "xvxg",
        ",": "xvxh",
    },
    encoded_prefix="_uwu_",
    decoded_strip_len=5,
)


def map_operators_to_strings(expr_str: str, mapping: OperatorMapping) -> str:
    expr_str = expr_str.replace(" ", "")
    for op, string in mapping.operator_to_string.items():
        expr_str = expr_str.replace(op, string)
    return mapping.encoded_prefix + expr_str


def map_strings_to_operators(expr_str: str, mapping: OperatorMapping) -> str:
    for op, string in mapping.string_to_operator.items():
        expr_str = expr_str.replace(op, string)
    return expr_str[mapping.decoded_strip_len:]


def create_sym_inputs(
    traced_graph: TracedGraph,
    size: List[Expr],
    operator_mapping: OperatorMapping,
) -> None:
    for s in size:
        if isinstance(s, (List, Tuple)):
            create_sym_inputs(traced_graph, s, operator_mapping)
            continue
        if isinstance(s, Expr) and not isinstance(s, Integer):
            s_name = str(s)
            if not isinstance(s, Symbol):
                s_name = map_operators_to_strings(s_name, operator_mapping)
            if s_name in traced_graph.sym_nodes:
                continue
            new_node = traced_graph.graph.placeholder(s_name)
            new_node.meta["val"] = V.graph.sizevars.shape_env.create_symintnode(s, hint=None)
            traced_graph.sym_nodes.update({s_name: new_node})


def process_ir_constant(
    inp: ExpandView,
    operator_mapping: OperatorMapping,
) -> tuple[Any, bool]:
    skip = False
    if isinstance(inp.data, IndexingConstant):
        dtype = inp.data.dtype
        inp = inp.data.index
        if dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if isinstance(inp, Expr) and not isinstance(inp, SympyNumber):
                traced_graph = TracedGraph()
                create_sym_inputs(traced_graph, [inp], operator_mapping)
                s_name = str(inp)
                if not isinstance(inp, Symbol):
                    s_name = map_operators_to_strings(str(inp), operator_mapping)
                traced_graph.last_node = traced_graph.sym_nodes[s_name]
                inp = traced_graph
            else:
                inp = float(inp)
    elif isinstance(inp.data, ir.Constant):
        inp = inp.data.value
    else:
        skip = True
    return inp, skip


def fetch_graphs(
    inputs: Optional[List[TensorBox]],
    operator_mapping: OperatorMapping,
    *,
    use_npu_meta: bool = False,
):
    if isinstance(inputs, (TensorBox, ir.StorageBox, ir.View, sympy.Symbol, ir.Constant, ir.ReinterpretView)):
        inputs = [inputs]
    input_graphs = []
    for inp in inputs:
        if isinstance(inp, List):
            input_graphs.append(fetch_graphs(inp, operator_mapping, use_npu_meta=use_npu_meta))
            continue
        if not isinstance(
            inp,
            (
                TensorBox,
                ir.StorageBox,
                ir.View,
                ir.ReinterpretView,
                ir.PermuteView,
                ir.SliceView,
                ir.ExpandView,
            ),
        ):
            input_graphs.append(inp)
            continue
        if isinstance(inp, ExpandView):
            inp, skip = process_ir_constant(inp, operator_mapping)
            if not skip:
                input_graphs.append(inp)
                continue
        name = inp.get_name()
        traced_graph = inp.get_traced_graph()
        if traced_graph is not None:
            input_graphs.append(traced_graph)
            continue
        traced_graph = TracedGraph()
        device = inp.get_device()
        dtype = inp.get_dtype()
        size = inp.get_size()
        stride = inp.get_stride()
        new_node = traced_graph.graph.placeholder(name)
        fake_input = create_fake_input(size, stride, device, dtype)
        new_node.meta["val"] = fake_input.npu() if use_npu_meta else fake_input
        traced_graph.last_node = new_node
        input_graphs.append(traced_graph)
    return input_graphs


def merge_traced_graphs(
    input_graphs: List[TracedGraph],
    origin_fn,
    node_name,
    operator_mapping: OperatorMapping,
    **kwargs,
):
    new_graph = TracedGraph()
    exist_nodes: Dict[str, torch.fx.Node] = {}

    def merge_graph(subgraphs: List[TracedGraph]) -> None:
        for input_graph in subgraphs:
            if isinstance(input_graph, List):
                merge_graph(input_graph)
                continue
            if not isinstance(input_graph, TracedGraph):
                continue
            for node in input_graph.graph.nodes:
                if node.name in exist_nodes:
                    continue
                new_node = new_graph.graph.node_copy(node, lambda n: exist_nodes.get(n.name, n))
                exist_nodes[node.name] = new_node
                if node.name in input_graph.sym_nodes:
                    new_graph.sym_nodes.update({node.name: new_node})

    def parse_args(subgraphs, nodes):
        args = []
        for input_graph in subgraphs:
            if isinstance(input_graph, TracedGraph):
                args.append(nodes[input_graph.last_node.name])
            elif isinstance(input_graph, (List, Tuple)):
                args.append(parse_args(input_graph, nodes))
            else:
                if isinstance(input_graph, Expr) and not isinstance(input_graph, Integer):
                    if not isinstance(input_graph, Symbol):
                        input_graph = map_operators_to_strings(str(input_graph), operator_mapping)
                    args.append(new_graph.sym_nodes[str(input_graph)])
                else:
                    args.append(input_graph)
        return args

    num_args = len(input_graphs)

    for k, v in kwargs.items():
        if isinstance(v, Expr) and not isinstance(v, Integer):
            traced_graph = TracedGraph()
            create_sym_inputs(traced_graph, [v], operator_mapping)
            s_name = str(v)
            if not isinstance(v, Symbol):
                s_name = map_operators_to_strings(str(v), operator_mapping)
            traced_graph.last_node = traced_graph.sym_nodes[s_name]
            kwargs[k] = traced_graph.sym_nodes[s_name]
            input_graphs.append(traced_graph)
    merge_graph(input_graphs)
    input_graphs = input_graphs[:num_args]
    create_sym_inputs(new_graph, input_graphs, operator_mapping)
    args = parse_args(input_graphs, exist_nodes)
    with new_graph.graph.inserting_after(new_graph.last_node):
        new_node = new_graph.graph.call_function(origin_fn, args=tuple(args), kwargs=kwargs)
    new_node.name = node_name
    new_graph.last_node = new_node
    return new_graph


def merge_fx_graphs(traced_graphs: List[TracedGraph]):
    new_graph = TracedGraph()
    exist_nodes: Dict[str, torch.fx.Node] = {}
    last_nodes = []

    def merge_graph(subgraphs: List[TracedGraph]) -> None:
        for input_graph in subgraphs:
            if isinstance(input_graph, List):
                merge_graph(input_graph)
                continue
            if not isinstance(input_graph, TracedGraph):
                continue
            for node in input_graph.graph.nodes:
                if node.name in exist_nodes:
                    continue
                new_node = new_graph.graph.node_copy(node, lambda n: exist_nodes.get(n.name, n))
                exist_nodes[node.name] = new_node
            last_nodes.append(exist_nodes[input_graph.last_node.name])

    merge_graph(traced_graphs)
    new_graph.last_node = last_nodes
    return new_graph


def subtract_graph(
    graph1: TracedGraph,
    graph2: TracedGraph,
    node_name=None,
) -> Tuple[TracedGraph, torch.fx.Node]:
    new_graph = TracedGraph()
    last_node2 = graph2.last_node
    graph1_node_names = {node.name for node in graph1.graph.nodes}
    graph2_node_names = {node.name for node in graph2.graph.nodes}
    placeholder = None
    exist_nodes: Dict[str, torch.fx.Node] = {}
    if node_name not in graph1_node_names:
        placeholder = new_graph.graph.placeholder(last_node2.name if node_name is None else node_name)
        exist_nodes[last_node2.name] = placeholder
    for node in graph1.graph.nodes:
        if node.name in graph2_node_names and node.name not in graph1.sym_nodes:
            continue
        new_node = new_graph.graph.node_copy(node, lambda n: exist_nodes.get(n.name, n))
        exist_nodes[node.name] = new_node
    new_graph.last_node = exist_nodes[graph1.last_node.name]
    new_graph.sym_nodes = graph1.sym_nodes
    return new_graph, placeholder
