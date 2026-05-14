# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FX Importer that converts a torch.fx.GraphModule into a torch-mlir Module (torch dialect).
"""

from collections import namedtuple

from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_importer import FxImporter

import torch
import torch.fx
from torch.export.graph_signature import (
    ConstantArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    TensorArgument,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges


__all__ = ["import_mlir_module_from_fx"]

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _normalize_var_to_range(var_to_range):
    """Convert torch symbolic infinities to finite i64 bounds for torch-mlir."""
    normalized = {}
    for symbol, value_range in var_to_range.items():
        if not getattr(value_range, "is_int", False):
            continue
        lower = _INT64_MIN if value_range.lower == -int_oo else value_range.lower
        upper = _INT64_MAX if value_range.upper == int_oo else value_range.upper
        normalized[symbol] = ValueRanges(lower, upper)
    return normalized


def import_mlir_module_from_fx(gm: torch.fx.GraphModule) -> ir.Module:
    """
    Imports a torch.fx.GraphModule into a torch-mlir Module (torch dialect).

    This function handles in-place mutation tracking and graph copying to ensure
    the original graph remains unmodified and safe for subsequent compilation steps.
    """
    # Create a copy of the graph to avoid in-place modification affecting the original graph
    new_graph = torch.fx.Graph()
    val_map = {}
    output_val = new_graph.graph_copy(gm.graph, val_map)
    new_graph.output(output_val)
    gm = torch.fx.GraphModule(gm, new_graph)

    torch_mlir_context = ir.Context()
    torch_d.register_dialect(torch_mlir_context)
    fx_importer = FxImporter(context=torch_mlir_context)

    # 1. Analyze inputs and initialize mutation tracking
    input_specs: list[InputSpec] = []
    # Map: placeholder_node -> placeholder_node (identity)
    tracked_placeholders = {}

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            arg = TensorArgument(name=node.name)
            if "val" in node.meta and isinstance(node.meta["val"], (torch.SymInt, int)):
                arg = SymIntArgument(name=node.name)
            input_specs.append(
                InputSpec(kind=InputKind.USER_INPUT, arg=arg, target=None)
            )
            tracked_placeholders[node] = node

    # 2. Analyze graph for mutations (in-place ops)
    # Map: node -> origin_placeholder_node
    node_to_origin = {n: n for n in tracked_placeholders}
    final_producers = {}

    for node in gm.graph.nodes:
        if node.op == "call_function":
            is_inplace = False
            # Check for standard inplace ops (convention: ends with '_')
            if hasattr(node.target, "__name__") and node.target.__name__.endswith("_"):
                is_inplace = True
            elif isinstance(node.target, torch._ops.OpOverload):
                if node.target.__name__.endswith("_"):
                    is_inplace = True
                elif hasattr(
                    node.target, "_schema"
                ) and node.target._schema.name.endswith("_"):
                    is_inplace = True

            # If it is an in-place op and the first argument is tracked
            if is_inplace and len(node.args) > 0:
                mutated_arg = node.args[0]
                if mutated_arg in node_to_origin:
                    origin = node_to_origin[mutated_arg]
                    # This node becomes the new "latest" value for the origin
                    final_producers[origin] = node
                    # Map this node to the origin so subsequent mutations can trace back
                    node_to_origin[node] = origin

    # Populate user_inputs_to_mutate: producer_name -> input_name
    user_inputs_to_mutate: dict[str, str] = {}
    for origin, producer in final_producers.items():
        user_inputs_to_mutate[producer.name] = origin.name

    # Transform in-place to functional
    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target == torch.ops.aten.copy_.default:
                node.target = torch.ops.aten.copy.default
            elif (
                getattr(node.target, "__name__", "") == "copy_"
                and getattr(node.target, "__module__", "") == "torch.ops.aten"
            ):
                if hasattr(torch.ops.aten, "copy"):
                    node.target = torch.ops.aten.copy.default

    # 3. Construct OutputSpecs (skip constant None outputs - torch-mlir import_program does not support USER_OUTPUT
    # ConstantArgument with value=None)
    output_specs: list[OutputSpec] = []
    output_node = next((n for n in gm.graph.nodes if n.op == "output"), None)
    if output_node:
        args = output_node.args[0]
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Filter out None outputs to avoid NotImplementedError in torch-mlir import_program
        valid_args = [a for a in args if a is not None]
        if len(valid_args) != len(args):
            output_node.args = (tuple(valid_args),)
            gm.recompile()

        for arg in valid_args:
            if isinstance(arg, torch.fx.Node):
                out_arg = TensorArgument(name=arg.name)
                if "val" in arg.meta and isinstance(
                    arg.meta["val"], (torch.SymInt, int)
                ):
                    out_arg = SymIntArgument(name=arg.name)
                output_specs.append(
                    OutputSpec(kind=OutputKind.USER_OUTPUT, arg=out_arg, target=None)
                )
            else:
                out_arg = ConstantArgument(
                    name=f"constant_output_{len(output_specs)}", value=arg
                )
                output_specs.append(
                    OutputSpec(kind=OutputKind.USER_OUTPUT, arg=out_arg, target=None)
                )

    FakeSignature = namedtuple(
        "FakeSignature",
        [
            "input_specs",
            "output_specs",
            "user_inputs_to_mutate",
        ],
    )

    sig = FakeSignature(
        input_specs=input_specs,
        output_specs=output_specs,
        user_inputs_to_mutate=user_inputs_to_mutate,
    )

    # Helper to get range constraints
    def _get_var_to_range(gm):
        shape_env = None
        for node in gm.graph.nodes:
            if "val" in node.meta:
                val = node.meta["val"]
                if hasattr(val, "fake_mode") and val.fake_mode is not None:
                    shape_env = val.fake_mode.shape_env
                    if shape_env is not None:
                        break
        if shape_env is None or not hasattr(shape_env, "var_to_range"):
            return {}
        return _normalize_var_to_range(shape_env.var_to_range)

    FakeExportedProgram = namedtuple(
        "FakeExportedProgram",
        ["graph", "graph_signature", "range_constraints"],
    )

    prog = FakeExportedProgram(
        graph=gm.graph,
        graph_signature=sig,
        range_constraints=_get_var_to_range(gm),
    )

    fx_importer.import_program(
        prog,
        func_name="main",
        import_symbolic_shape_expressions=True,
    )
    return fx_importer.module
