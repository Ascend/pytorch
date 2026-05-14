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
FX Exporter that converts a torch-mlir Module (in the 'torch' dialect) back into a
torch.fx.GraphModule.
"""

import logging
import operator
import re
from typing import Any

from torch_mlir import ir
from torch_mlir.extras.fx_importer import (
    TORCH_DTYPE_TO_INT,
    TORCH_LAYOUT_TO_INT,
    TORCH_MEMORY_FORMAT_TO_INT,
)

import torch
import torch.fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_npu._inductor.mfusion import subgraph_registry


__all__ = [
    "FxExporter",
    "export_mlir_module_to_fx",
    "fake_tensor_propagate_mfusion_subgraph",
]

logger = logging.getLogger(__name__)

# Process-local: torch.library.custom_op(full_op_name) must run once per library key.
_REGISTERED_SYMBOL_SUBGRAPH_CUSTOM_OPS: set[str] = set()

aten = torch.ops.aten


def _flat_symbol_ref_attr_to_symbol_name(attr: ir.Attribute) -> str:
    """Return the referenced symbol name without a leading ``@``.

    Prefer MLIR's ``FlatSymbolRefAttr.value`` instead of ``str(attr)`` so we do not
    depend on textual forms that may include quotes or dialect-specific prefixes.
    """
    FlatSymbolRefAttr = getattr(ir, "FlatSymbolRefAttr", None)
    if FlatSymbolRefAttr is not None:
        try:
            fsr = FlatSymbolRefAttr(attr)
            return str(fsr.value).strip()
        except (ValueError, TypeError, AttributeError):
            try:
                if isinstance(attr, FlatSymbolRefAttr):
                    return str(attr.value).strip()
            except TypeError:
                pass

    s = str(attr).strip().strip('"').strip("'").strip()
    if s.startswith("@"):
        s = s[1:].strip()
    return s


def _infer_aten_mm_2d_output_shape(
    lhs_shape: tuple[Any, ...], rhs_shape: tuple[Any, ...], trans_a: bool, trans_b: bool
) -> tuple[Any, Any]:
    """2D matmul output shape matching mfuse.aclnn.mm trans_x1/trans_x2 (last-two-dim transpose)."""
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise NotImplementedError(
            "dvm mm fake shape inference only supports 2D operands; "
            f"got lhs {lhs_shape}, rhs {rhs_shape}"
        )
    l0, l1 = lhs_shape
    r0, r1 = rhs_shape
    el0, el1 = (l1, l0) if trans_a else (l0, l1)
    er0, er1 = (r1, r0) if trans_b else (r0, r1)
    # (el0, el1) @ (er0, er1) with el1 == er0
    return (el0, er1)


def _make_dvm_mm_fake_forward(trans_a: bool, trans_b: bool):
    """Returns a 2-arg callable that allocates the correct FakeTensor output for aclnn-style mm."""

    def _fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        shape = _infer_aten_mm_2d_output_shape(
            tuple(a.shape), tuple(b.shape), trans_a, trans_b
        )
        return a.new_empty(shape)

    return _fn


def patch_dvm_mm_targets_for_fake_eval(
    gm: torch.fx.GraphModule,
) -> list[tuple[torch.fx.Node, Any]]:
    """Temporarily replace aten.mm with a shape-only stub where dvm_trans_* is set (storage layout != torch.mm)."""
    patches: list[tuple[torch.fx.Node, Any]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not aten.mm.default:
            continue
        if not (node.meta.get("dvm_trans_a") or node.meta.get("dvm_trans_b")):
            continue
        ta = bool(node.meta.get("dvm_trans_a", False))
        tb = bool(node.meta.get("dvm_trans_b", False))
        patches.append((node, node.target))
        node.target = _make_dvm_mm_fake_forward(ta, tb)
    if patches:
        gm.recompile()
    return patches


def restore_dvm_mm_patches(
    gm: torch.fx.GraphModule, patches: list[tuple[torch.fx.Node, Any]]
) -> None:
    for node, target in patches:
        node.target = target
    if patches:
        gm.recompile()


def fake_tensor_propagate_mfusion_subgraph(
    sub_gm: torch.fx.GraphModule,
    fake_inputs: list,
    fake_mode: FakeTensorMode | None = None,
) -> None:
    """FakeTensorProp on an MFusion-exported subgraph; fixes mm when dvm_trans_* disagrees with torch.mm.

    MFusion convert-mfuse-to-torch attaches ``dvm_trans_a`` / ``dvm_trans_b`` on ``torch.aten.mm`` for
    fused transpose+matmul; storage shapes of operands still match the *untransposed* layout, so plain
    ``aten.mm`` FakeTensor meta is inconsistent. We temporarily replace ``mm`` with a shape-only stub
    during propagation (see ``patch_dvm_mm_targets_for_fake_eval``).
    """
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    patches = patch_dvm_mm_targets_for_fake_eval(sub_gm)
    try:
        if fake_mode is not None:
            with fake_mode:
                FakeTensorProp(sub_gm, mode=fake_mode).propagate(*fake_inputs)
        else:
            FakeTensorProp(sub_gm).propagate(*fake_inputs)
    finally:
        restore_dvm_mm_patches(sub_gm, patches)


# Reverse mappings for enums for converting integer flags back to torch types.
INT_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_INT.items()}
INT_TO_TORCH_LAYOUT = {v: k for k, v in TORCH_LAYOUT_TO_INT.items()}
INT_TO_TORCH_MEMORY_FORMAT = {v: k for k, v in TORCH_MEMORY_FORMAT_TO_INT.items()}


def _get_python_value_from_attr(attr: ir.Attribute) -> Any:
    """Extracts a Python primitive value from an MLIR attribute."""
    if isinstance(attr, ir.IntegerAttr):
        return attr.value
    elif isinstance(attr, ir.FloatAttr):
        return attr.value
    elif isinstance(attr, ir.StringAttr):
        return attr.value
    elif isinstance(attr, ir.BoolAttr):
        return attr.value
    else:
        raise NotImplementedError(f"unknown attr type: {type(attr)}")


def _get_schema_type_from_mlir_type(mlir_type: ir.Type) -> str:
    """Converts an MLIR type to a schema string representation."""
    mlir_type_str = str(mlir_type)
    if mlir_type_str in ("!torch.vtensor", "!torch.tensor"):
        return "Tensor"
    elif mlir_type_str in ("!torch.str", "!torch.string"):
        return "str"
    elif mlir_type_str == "!torch.int":
        return "int"
    elif mlir_type_str == "!torch.float":
        return "float"
    else:
        return "Any"


def _get_schema_from_func_op(func_op: ir.Operation) -> str:
    """Constructs a function schema string from an MLIR func op."""
    func_type = ir.FunctionType(ir.TypeAttr(func_op.attributes["function_type"]).value)

    arg_types = func_type.inputs
    ret_types = func_type.results

    arg_strs = [
        f"{_get_schema_type_from_mlir_type(t)} arg{i}" for i, t in enumerate(arg_types)
    ]
    args_str = ", ".join(arg_strs)

    if len(ret_types) == 1:
        ret_str = _get_schema_type_from_mlir_type(ret_types[0])
    else:
        ret_str = (
            f"({', '.join([_get_schema_type_from_mlir_type(t) for t in ret_types])})"
        )

    return f"({args_str}) -> {ret_str}"


def _get_schema_from_operator(op: ir.Operation) -> str:
    """Constructs a schema string from a torch.operator op."""
    arg_types = [v.type for v in op.operands]
    ret_types = [v.type for v in op.results]

    arg_strs = [
        f"{_get_schema_type_from_mlir_type(t)} arg{i}" for i, t in enumerate(arg_types)
    ]
    args_str = ", ".join(arg_strs)

    if len(ret_types) == 1:
        ret_str = _get_schema_type_from_mlir_type(ret_types[0])
    else:
        ret_str = (
            f"({', '.join([_get_schema_type_from_mlir_type(t) for t in ret_types])})"
        )

    return f"({args_str}) -> {ret_str}"


def _tensor_from_mlir_type(mlir_type: ir.Type) -> torch.Tensor | None:
    type_str = str(mlir_type)
    match = re.match(r"!torch\.(?:vtensor|tensor)<\[(.*?)\],\s*([a-z0-9]+)>", type_str)
    if match is None:
        return None
    shape_str, dtype_str = match.groups()
    dims: list[int] = []
    for dim in shape_str.split(","):
        dim = dim.strip()
        if not dim or dim == "?" or not dim.lstrip("-").isdigit():
            dims.append(1)
        else:
            dims.append(int(dim))

    dtype_map = {
        "f16": torch.float16,
        "f32": torch.float32,
        "f64": torch.float64,
        "bf16": torch.bfloat16,
        "i64": torch.int64,
        "i32": torch.int32,
        "i8": torch.int8,
        "ui8": torch.uint8,
        "i1": torch.bool,
    }
    dtype = dtype_map.get(dtype_str)
    if dtype is None:
        return None
    return torch.empty(dims, dtype=dtype)


def _attach_placeholder_meta_from_func(
    gm: torch.fx.GraphModule, func_op: ir.Operation
) -> None:
    func_type = ir.FunctionType(ir.TypeAttr(func_op.attributes["function_type"]).value)
    arg_types = list(func_type.inputs)
    arg_index = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if arg_index >= len(arg_types):
            break
        val = _tensor_from_mlir_type(arg_types[arg_index])
        if val is not None:
            node.meta["val"] = val
        arg_index += 1


def _build_fake_inputs(gm: torch.fx.GraphModule) -> list[Any] | None:
    fake_mode = FakeTensorMode()
    args: list[Any] = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            args.append(fake_mode.from_tensor(val))
        else:
            args.append(val)
    return args


def _propagate_fake_meta(gm: torch.fx.GraphModule) -> None:
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    args = _build_fake_inputs(gm)
    if args is None:
        return
    FakeTensorProp(gm).propagate(*args)


def _get_op_target(op_name: str) -> Any:
    """Resolves a torch.ops target from an MLIR op name string."""
    # Convert 'torch.aten.add.Tensor' -> torch.ops.aten.add.Tensor
    if not op_name.startswith("torch."):
        raise NotImplementedError(f"unknown op: {op_name}")

    name = op_name[6:]  # Strip "torch."
    parts = name.split(".")

    # Traverse torch.ops
    try:
        curr = torch.ops
        for p in parts:
            curr = getattr(curr, p)

        # If we got a packet (e.g. torch.ops.aten.expand), try to get .default
        if isinstance(curr, torch._ops.OpOverloadPacket) and hasattr(curr, "default"):
            return curr.default

        return curr
    except AttributeError:
        raise NotImplementedError(f"unknown op: {op_name}") from None


def _convert_arg_by_schema(arg_value: Any, schema_arg_type: Any, arg_name: str) -> Any:
    """Converts argument values based on the expected schema type (e.g. enum conversion)."""
    schema_type_str = str(schema_arg_type)
    if (
        arg_name == "dtype" or "ScalarType" in schema_type_str
    ):  # internal repr for dtype
        if isinstance(arg_value, int) and arg_value in INT_TO_TORCH_DTYPE:
            return INT_TO_TORCH_DTYPE[arg_value]
    elif arg_name == "layout" or "Layout" in schema_type_str:
        if isinstance(arg_value, int) and arg_value in INT_TO_TORCH_LAYOUT:
            return INT_TO_TORCH_LAYOUT[arg_value]
    elif arg_name == "memory_format" or "MemoryFormat" in schema_type_str:
        if isinstance(arg_value, int) and arg_value in INT_TO_TORCH_MEMORY_FORMAT:
            return INT_TO_TORCH_MEMORY_FORMAT[arg_value]
    elif arg_name == "device" or "Device" in schema_type_str:
        if isinstance(arg_value, str):
            return torch.device(arg_value)
    elif schema_type_str in ["int[]", "SymInt[]", "List[int]"]:
        if isinstance(arg_value, list):
            return tuple(arg_value)
    return arg_value


class FxExporter:
    """Exports an MLIR module (torch dialect) to a torch.fx.GraphModule."""

    def __init__(self, export_single_tuple_output: bool = True):
        self.export_single_tuple_output = export_single_tuple_output
        self.graph = torch.fx.Graph()
        self.value_map: dict[
            ir.Value, Any
        ] = {}  # Maps MLIR Value to FX Node or Python Value
        self.constants_module = torch.nn.Module()
        self.module: ir.Module | None = None

    def export(self, module: ir.Module) -> torch.fx.GraphModule:
        """
        Export an MLIR module to a torch.fx.GraphModule.
        Assumes the module contains a 'main' function or takes the first function found.
        """
        self.module = module
        # Find the main function
        main_func = None
        for op in module.body.operations:
            op_name = op.operation.name if hasattr(op, "operation") else op.name
            if op_name == "func.func":
                # Check symbol name
                if op.attributes["sym_name"].value == "main":
                    main_func = op
                    break

        if main_func is None:
            # Fallback: take the first func
            for op in module.body.operations:
                op_name = op.operation.name if hasattr(op, "operation") else op.name
                if op_name == "func.func":
                    main_func = op
                    break

        if main_func is None:
            raise ValueError("No function found in MLIR module")

        return self.export_func(main_func)

    def export_func(self, func_op: ir.Operation) -> torch.fx.GraphModule:
        """Export a specific MLIR FuncOp to a torch.fx.GraphModule."""
        self._process_func(func_op)
        return torch.fx.GraphModule(self.constants_module, self.graph)

    def _process_func(self, func_op: ir.Operation):
        # We assume single block function for now (functionalized)
        if len(func_op.regions[0].blocks) != 1:
            raise ValueError("Only single-block functions are supported.")

        block = func_op.regions[0].blocks[0]

        # Process arguments
        for i, arg in enumerate(block.arguments):
            node = self.graph.placeholder(f"arg{i}")
            self.value_map[arg] = node

        # Process operations
        for op in block.operations:
            self._process_op(op)

    def _process_op(self, op: ir.Operation):
        # Use op.operation.name to be safe against attribute shadowing
        op_name = op.operation.name if hasattr(op, "operation") else op.name

        if op_name.startswith("torch.constant."):
            self._process_constant(op)
        elif op_name.startswith(("torch.aten.", "torch.prims.")):
            self._process_aten_op(op)
        elif op_name == "torch.operator":
            self._process_operator(op)
        elif op_name == "func.return":
            self._process_return(op)
        elif op_name == "torch.prim.ListConstruct":
            self._process_list_construct(op)
        elif op_name == "torch.prim.ListUnpack":
            self._process_list_unpack(op)
        elif op_name == "torch.prim.TupleConstruct":
            self._process_tuple_construct(op)
        elif op_name == "torch.prim.TupleUnpack":
            self._process_tuple_unpack(op)
        elif op_name == "torch.prim.NumToTensor.Scalar":
            self._process_num_to_tensor(op)
        elif op_name == "torch.copy.to_vtensor":
            self._process_copy_to_vtensor(op)
        elif op_name == "torch.overwrite.tensor.contents":
            self._process_overwrite_tensor_contents(op)
        elif op_name in ("torch.symbolic_int", "torch.bind_symbolic_shape"):
            pass  # Ignore symbolic ops
        else:
            raise NotImplementedError(f"unknown op: {op_name}")

    def _process_copy_to_vtensor(self, op: ir.Operation):
        # Pass-through for value tensor conversion
        val = self.value_map[op.operands[0]]
        self.value_map[op.results[0]] = val

    def _process_overwrite_tensor_contents(self, op: ir.Operation):
        # Convert back to copy_ (inplace copy)
        source = self.value_map[op.operands[0]]
        dest = self.value_map[op.operands[1]]
        self.graph.call_function(torch.ops.aten.copy_.default, (dest, source, False))

    def _process_constant(self, op: ir.Operation):
        op_name = op.operation.name if hasattr(op, "operation") else op.name
        if op_name == "torch.constant.none":
            val = None
        else:
            val = _get_python_value_from_attr(op.attributes["value"])
        self.value_map[op.results[0]] = val

    def _process_list_construct(self, op: ir.Operation):
        inputs = [self.value_map[arg] for arg in op.operands]
        self.value_map[op.results[0]] = inputs

    def _process_list_unpack(self, op: ir.Operation):
        input_list = self.value_map[op.operands[0]]
        if isinstance(input_list, (list, tuple)):
            for i, result in enumerate(op.results):
                self.value_map[result] = input_list[i]
        else:
            for i, result in enumerate(op.results):
                node = self.graph.call_function(operator.getitem, (input_list, i))
                self.value_map[result] = node

    def _process_tuple_construct(self, op: ir.Operation):
        inputs = tuple(self.value_map[arg] for arg in op.operands)
        self.value_map[op.results[0]] = inputs

    def _process_tuple_unpack(self, op: ir.Operation):
        self._process_list_unpack(op)

    def _process_num_to_tensor(self, op: ir.Operation):
        input_val = self.value_map[op.operands[0]]
        node = self.graph.call_function(torch.as_tensor, (input_val,))
        self.value_map[op.results[0]] = node

    def _process_return(self, op: ir.Operation):
        args = [self.value_map[arg] for arg in op.operands]
        if len(args) == 0:
            self.graph.output(None)
        elif len(args) == 1 and not self.export_single_tuple_output:
            self.graph.output(args[0])
        else:
            self.graph.output(tuple(args))

    def _process_operator(self, op: ir.Operation):
        target_name = op.attributes["name"].value
        has_mlir_attr = "mfusion.subgraph_mlir" in op.attributes
        has_dynamic_attr = "mfusion.is_dynamic" in op.attributes
        has_subgraph_symbol = "subgraph" in op.attributes
        if has_mlir_attr or has_dynamic_attr:
            if not (has_mlir_attr and has_dynamic_attr):
                raise ValueError(
                    f"Operator {target_name} missing required mfusion attributes"
                )
            mapped_operands = [self.value_map[arg] for arg in op.operands]
            if not mapped_operands:
                raise ValueError(f"Operator {target_name} missing operands")
            subgraph_name = mapped_operands[-1]
            logger.debug("op name: %s, subgraph name: %s", target_name, subgraph_name)
            self._register_fused_operator(op, target_name, subgraph_name)
        elif has_subgraph_symbol:
            # Generic fused-op path used by fx_mlir_converter UT:
            # torch.operator "torch.fused.*" {subgraph = @func_symbol}
            self._register_symbol_subgraph_operator(op, target_name)

        target = _get_op_target(target_name)
        self._process_aten_op_with_target(op, target)

    def _register_symbol_subgraph_operator(
        self, op: ir.Operation, target_name: str
    ) -> None:
        """Register torch.operator target from a plain `subgraph = @sym` reference.

        This path is intentionally lightweight and used when operator carries only
        a symbol reference (without mfusion.subgraph_mlir / mfusion.is_dynamic).
        It keeps fx_mlir_converter roundtrip tests runnable in eager backends.
        """
        if self.module is None:
            raise RuntimeError("Module not set in FxExporter")

        subgraph_attr = op.attributes["subgraph"]
        subgraph_name = _flat_symbol_ref_attr_to_symbol_name(subgraph_attr)
        if not subgraph_name:
            raise ValueError(f"Operator {target_name} has invalid subgraph symbol")

        sub_func = None
        for o in self.module.body.operations:
            op_name = o.operation.name if hasattr(o, "operation") else o.name
            if (
                op_name == "func.func"
                and o.attributes["sym_name"].value == subgraph_name
            ):
                sub_func = o
                break
        if sub_func is None:
            raise ValueError(f"Subgraph {subgraph_name} not found")

        # Convert torch.fused.mul_add -> fused::mul_add
        parts = target_name.split(".")
        if len(parts) < 3:
            raise ValueError(f"Malformed fused op name: {target_name}")
        lib_name = parts[1]
        op_leaf = "_".join(parts[2:])
        full_op_name = f"{lib_name}::{op_leaf}"

        try:
            _get_op_target(target_name)
            return
        except NotImplementedError:
            pass

        if full_op_name in _REGISTERED_SYMBOL_SUBGRAPH_CUSTOM_OPS:
            try:
                _get_op_target(target_name)
                return
            except NotImplementedError as e:
                raise RuntimeError(
                    f"Custom op {full_op_name!r} was registered earlier in this process "
                    f"but {target_name!r} is still not resolvable on torch.ops"
                ) from e

        sub_exporter = FxExporter()
        sub_gm = sub_exporter.export_func(sub_func)
        schema = _get_schema_from_operator(op)

        try:

            @torch.library.custom_op(full_op_name, mutates_args=(), schema=schema)
            def _symbol_subgraph_impl(*args):
                out = sub_gm(*args)
                return out

        except (RuntimeError, ValueError) as e:
            msg = str(e).lower()
            if any(
                x in msg
                for x in (
                    "duplicate",
                    "already",
                    "exists",
                    "re-register",
                    "re_register",
                )
            ):
                _REGISTERED_SYMBOL_SUBGRAPH_CUSTOM_OPS.add(full_op_name)
                try:
                    _get_op_target(target_name)
                    return
                except NotImplementedError as e2:
                    raise RuntimeError(
                        f"torch.library reported a duplicate for {full_op_name!r} but "
                        f"{target_name!r} is still not resolvable"
                    ) from e2
            raise
        else:
            _REGISTERED_SYMBOL_SUBGRAPH_CUSTOM_OPS.add(full_op_name)

    def _register_fused_operator(
        self, op: ir.Operation, target_name: str, subgraph_name: str
    ):
        if not isinstance(subgraph_name, str) or not subgraph_name.strip():
            raise ValueError(
                f"Operator {target_name} missing non-empty subgraph_name operand"
            )

        if "mfusion.subgraph_mlir" in op.attributes:
            subgraph_mlir_attr = op.attributes["mfusion.subgraph_mlir"]
        else:
            subgraph_mlir_attr = None
        if subgraph_mlir_attr is None:
            raise ValueError(
                f"Operator {target_name} missing 'subgraph_mlir' attribute"
            )
        subgraph_mlir = subgraph_mlir_attr.value
        if not isinstance(subgraph_mlir, str) or not subgraph_mlir.strip():
            raise ValueError(
                f"Operator {target_name} has empty 'subgraph_mlir' attribute"
            )

        if "mfusion.is_dynamic" in op.attributes:
            is_dynamic_attr = op.attributes["mfusion.is_dynamic"]
        else:
            is_dynamic_attr = None
        if is_dynamic_attr is None:
            raise ValueError(f"Operator {target_name} missing 'is_dynamic' attribute")
        is_dynamic = is_dynamic_attr.value
        if not isinstance(is_dynamic, bool):
            raise TypeError(
                f"Operator {target_name} has non-bool 'is_dynamic' attribute"
            )

        if self.module is None:
            raise RuntimeError("Module not set in FxExporter")

        sub_func = None
        for o in self.module.body.operations:
            op_name = o.operation.name if hasattr(o, "operation") else o.name
            if (
                op_name == "func.func"
                and o.attributes["sym_name"].value == subgraph_name
            ):
                sub_func = o
                break

        if sub_func is None:
            raise ValueError(f"Subgraph {subgraph_name} not found")

        # torch.fused.mul_add -> fused::mul_add
        parts = target_name.split(".")
        if len(parts) < 3:
            raise ValueError(f"Malformed fused op name: {target_name}")

        lib_name = parts[1]
        op_name = "_".join(parts[2:])
        full_op_name = f"{lib_name}::{op_name}"

        # Recursively export subgraph
        sub_exporter = FxExporter(export_single_tuple_output=False)
        sub_gm = sub_exporter.export_func(sub_func)
        # _attach_placeholder_meta_from_func(sub_gm, sub_func)
        # _propagate_fake_meta(sub_gm)
        schema = _get_schema_from_operator(op)

        payload = subgraph_registry.Payload(
            fx_gm=sub_gm,
            mlir=subgraph_mlir,
            is_dynamic=is_dynamic,
            torch_name=target_name,
            full_op_name=full_op_name,
        )
        subgraph_registry.register(subgraph_name, payload)

        # Register custom op dynamically (if not already registered)
        try:
            _get_op_target(target_name)
            return
        except NotImplementedError:
            pass

        @torch.library.custom_op(full_op_name, mutates_args=(), schema=schema)
        def _fused_impl(*args):
            raise RuntimeError(
                "mfusion custom_op should not run in eager mode. "
                "It must be lowered by inductor to dvm/akg kernel execution."
            )

        @_fused_impl.register_fake
        def _(*args):
            if not args:
                raise RuntimeError("mfusion custom_op missing subgraph_name")
            subgraph_name = args[-1]
            if not isinstance(subgraph_name, str) or not subgraph_name.strip():
                raise RuntimeError("mfusion custom_op has invalid subgraph_name")
            payload = subgraph_registry.get(subgraph_name)
            gm = payload.fx_gm
            patches = patch_dvm_mm_targets_for_fake_eval(gm)
            try:
                return gm(*args[:-1])
            finally:
                restore_dvm_mm_patches(gm, patches)

    def _process_aten_op(self, op: ir.Operation):
        op_name = op.operation.name if hasattr(op, "operation") else op.name

        # Handle special case for copy
        if op_name == "torch.aten.copy":
            # In MLIR, torch.aten.copy returns a new tensor. In FX/PyTorch, it's typically used in-place or as copy_
            # Here we treat it as passing through the new value (arg1).
            # arg0 is destination, arg1 is source, arg2 is non_blocking
            self.value_map[op.results[0]] = self.value_map[op.operands[1]]
            return

        target = _get_op_target(op_name)
        self._process_aten_op_with_target(op, target)

    def _trim_default_args(self, pos_args_info):
        """Trims trailing arguments that match their default values."""
        while pos_args_info:
            val, default_val, has_default = pos_args_info[-1]
            if has_default and val == default_val:
                pos_args_info.pop()
            else:
                break
        return tuple(x[0] for x in pos_args_info)

    def _process_aten_op_with_target(self, op: ir.Operation, target: Any):
        args = []
        kwargs = {}

        schema = getattr(target, "_schema", None)
        logger.debug("schema: %s", schema)
        schema_args = schema.arguments if schema else []

        mapped_operands = [self.value_map[arg] for arg in op.operands]

        if schema:
            # Temporary list to hold positional args and their defaults
            # (value, default_value, has_default)
            pos_args_info = []

            for i, arg_val in enumerate(mapped_operands):
                if i < len(schema_args):
                    arg_schema = schema_args[i]
                    arg_type = arg_schema.type
                    val = _convert_arg_by_schema(arg_val, arg_type, arg_schema.name)

                    if arg_schema.kwarg_only:
                        # Only add kwarg if it's not the default value.
                        # This avoids issues with inductor lowerings that don't expect
                        # certain optional arguments even if they are in the schema.
                        has_default = False
                        if hasattr(arg_schema, "has_default_value"):
                            has_default = arg_schema.has_default_value()

                        if not (has_default and val == arg_schema.default_value):
                            kwargs[arg_schema.name] = val
                    else:
                        # Check for default value support
                        has_default = False
                        if hasattr(arg_schema, "has_default_value"):
                            has_default = arg_schema.has_default_value()

                        pos_args_info.append(
                            (val, arg_schema.default_value, has_default)
                        )
                else:
                    # Extra args (varargs?), assume no default to be safe
                    pos_args_info.append((arg_val, None, False))

            args = self._trim_default_args(pos_args_info)
        else:
            args = tuple(mapped_operands)

        # Debug: log args before creating FX node
        if "mfusion" in str(target):
            for i, arg in enumerate(args):
                logger.debug(
                    "Creating FX node for mfusion op - arg %d: %r, type: %s",
                    i,
                    arg,
                    type(arg),
                )

        node = self.graph.call_function(target, args, kwargs)

        # Transpose flags from convert-mfuse-to-torch (mfuse.aclnn.mm trans_x1/trans_x2 -> discardable attrs).
        if target is torch.ops.aten.mm.default:
            for name in ("dvm_trans_a", "dvm_trans_b"):
                if name in op.attributes:
                    node.meta[name] = bool(
                        _get_python_value_from_attr(op.attributes[name])
                    )

        if len(op.results) == 1:
            self.value_map[op.results[0]] = node
        elif len(op.results) > 1:
            for i, res in enumerate(op.results):
                getitem_node = self.graph.call_function(operator.getitem, (node, i))
                self.value_map[res] = getitem_node
        else:
            # Op has no results (e.g. side-effect only), node is already created in graph
            pass


def export_mlir_module_to_fx(module: ir.Module) -> torch.fx.GraphModule:
    """Wrapper function to export an MLIR module to FX."""
    exporter = FxExporter()
    return exporter.export(module)
