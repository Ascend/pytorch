# Copyright (c) 2026, Huawei Technologies Co., Ltd

"""C++ wrapper support for the isolated Triton NPU backend."""

import ast
import dataclasses
from typing import Any
from collections.abc import Mapping

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codegen.cpp_wrapper_gpu import DeferredTritonCallWrapper

from torch_npu._inductor.codegen.cpp_wrapper_npu import (
    CppWrapperNpu,
    DeferredNpuTritonCallWrapper,
)

from .. import device_props
from ..launcher_codegen import _can_clamp_1d_grid


# Preserve tensor strides when casting through the JIT/AOT tensor-handle ABI.
_DOWNCAST_HELPERS = """
static inline int64_t te_downcast_span(AtenTensorHandle tensor) {
    int64_t ndim = 0;
    int64_t* sizes = nullptr;
    int64_t* strides = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(tensor, &ndim));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(tensor, &sizes));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(tensor, &strides));
    int64_t span = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (sizes[d] == 0) return 0;
        span += strides[d] * (sizes[d] - 1);
    }
    return span;
}

static inline RAIIAtenTensorHandle te_downcast_flat(AtenTensorHandle tensor) {
    int64_t span = te_downcast_span(tensor);
    int64_t stride = 1;
    return RAIIAtenTensorHandle(
        reinterpret_tensor_wrapper(tensor, 1, &span, &stride, 0));
}

static inline RAIIAtenTensorHandle te_downcast(
        AtenTensorHandle tensor, int32_t dtype, bool preserve_input) {
    int64_t ndim = 0;
    int64_t* sizes = nullptr;
    int64_t* strides = nullptr;
    int32_t device_type = 0;
    int32_t device_index = 0;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(tensor, &ndim));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(tensor, &sizes));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(tensor, &strides));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(tensor, &device_type));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_index(tensor, &device_index));
    AtenTensorHandle raw = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
        ndim, sizes, strides, dtype, device_type, device_index, &raw));
    RAIIAtenTensorHandle result(raw);
    if (preserve_input) {
        auto src = te_downcast_flat(tensor);
        auto dst = te_downcast_flat(result);
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(dst, src, false));
    }
    return result;
}

static inline void te_downcast_writeback(
        AtenTensorHandle original, AtenTensorHandle temporary) {
    int64_t ndim = 0;
    int64_t* sizes = nullptr;
    int64_t* strides = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(original, &ndim));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(original, &sizes));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(original, &strides));
    for (int64_t d = 0; d < ndim; ++d) {
        if (sizes[d] > 1 && strides[d] == 0) {
            auto dst = te_downcast_flat(original);
            auto src = te_downcast_flat(temporary);
            AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(dst, src, false));
            return;
        }
    }
    // Logical copy leaves the unrelated cells in a gapped destination intact.
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_copy_(original, temporary, false));
}
"""


def _config_kwargs(config: Any) -> dict[str, Any]:
    return dict(config.kwargs) if hasattr(config, "kwargs") else dict(config)


def _expr_to_cpp(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return str(int(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operator = "+" if isinstance(node.op, ast.UAdd) else "-"
        return f"({operator}{_expr_to_cpp(node.operand)})"
    if isinstance(node, ast.BinOp):
        operator = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.FloorDiv: "/",
            ast.Mod: "%",
        }.get(type(node.op))
        if operator:
            return f"({_expr_to_cpp(node.left)} {operator} {_expr_to_cpp(node.right)})"
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        operator = {
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }.get(type(node.ops[0]))
        if operator:
            return f"({_expr_to_cpp(node.left)} {operator} {_expr_to_cpp(node.comparators[0])})"
    if isinstance(node, ast.IfExp):
        return (
            f"({_expr_to_cpp(node.test)} ? {_expr_to_cpp(node.body)} : "
            f"{_expr_to_cpp(node.orelse)})"
        )
    raise ValueError(f"unsupported grid-recipe expression: {ast.dump(node)}")


def _recipe_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _emit_recipe(prefix, recipe: Mapping[str, Any], declared: set[str]) -> None:
    lines, factors = recipe.get("lines"), recipe.get("factors")
    if not isinstance(lines, (list, tuple)) or not isinstance(factors, (list, tuple)):
        raise ValueError("NPU dispatch recipe requires lines and factors")
    if not factors or any(
        not isinstance(item, str) or not item.strip() for item in [*lines, *factors]
    ):
        raise ValueError("NPU dispatch recipe contains an empty statement")

    for source in lines:
        statement = ast.parse(source, mode="exec").body
        if len(statement) != 1 or not isinstance(statement[0], ast.Assign):
            raise ValueError(f"unsupported grid-recipe statement: {source}")
        assignment = statement[0]
        if len(assignment.targets) != 1 or not isinstance(
            assignment.targets[0], ast.Name
        ):
            raise ValueError(f"unsupported grid-recipe target: {source}")
        unknown = _recipe_names(assignment.value) - declared
        if unknown:
            raise ValueError(
                f"grid-recipe references undefined names: {', '.join(sorted(unknown))}"
            )
        name = assignment.targets[0].id
        expression = _expr_to_cpp(assignment.value)
        if "block" in name.lower() or "tile" in name.lower():
            expression = f"std::max<int64_t>(1, {expression})"
        prefix.writeline(
            f"{'int64_t ' if name not in declared else ''}{name} = {expression};"
        )
        declared.add(name)

    expressions = []
    for factor in factors:
        expression = ast.parse(factor, mode="eval").body
        unknown = _recipe_names(expression) - declared
        if unknown:
            raise ValueError(
                f"grid-recipe factor references undefined names: {', '.join(sorted(unknown))}"
            )
        expressions.append(_expr_to_cpp(expression))
    prefix.writeline(
        "uint32_t grid_0 = static_cast<uint32_t>(std::max<int64_t>(1, "
        + " * ".join(expressions)
        + "));"
    )


def _emit_grid(prefix, params: Mapping[str, Any]) -> None:
    metadata = params["inductor_meta"]
    def_args = params["def_args"]
    signature = params.get("triton_meta", {}).get("signature", {})
    runtime_names = set(def_args) | set(params.get("runtime_blocks", {}))
    arg_names = runtime_names | set(signature)
    recipe = metadata.get("npu_dispatch_recipe")
    num_cores = device_props.get_npu_vector_core_count()
    uses_recipe = (
        device_props.is_a5() and bool(recipe) and metadata.get("te_combo_meta") is None
    )

    if metadata.get("te_combo_meta") is not None:
        grid = str(num_cores)
    elif uses_recipe:
        # Signature-only constexpr arguments are not declared C++ variables.
        names = set(runtime_names)
        for name, value in _config_kwargs(params["config"]).items():
            if name.endswith("BLOCK") and name not in names:
                prefix.writeline(f"constexpr int64_t {name} = {value};")
                names.add(name)
        _emit_recipe(prefix, recipe, names)
    elif (
        metadata.get("npu_linearize", True)
        and metadata.get("npu_num_x_nodes", 0) == 0
        and metadata.get("grid_type", "Grid1D") == "Grid1D"
        and "R0_BLOCK" in arg_names
        and not metadata.get("npu_rsplit_partial", "ws_ptr" in arg_names)
    ):
        grid = "1"
    elif _can_clamp_1d_grid(metadata, def_args, arg_names):
        xblock = _config_kwargs(params["config"]).get("XBLOCK", 1)
        grid = f"std::max<int64_t>(1, std::min<int64_t>((xnumel + {xblock} - 1) / {xblock}, {num_cores}))"
    elif (
        not metadata.get("npu_linearize", True)
        and metadata.get("grid_type", "Grid1D") == "Grid1D"
        and "xnumel" in def_args
    ):
        xblock = _config_kwargs(params["config"]).get("XBLOCK", 1)
        grid = f"std::max<int64_t>(1, (xnumel + {xblock} - 1) / {xblock})"
    else:
        grid = str(num_cores)

    if not uses_recipe:
        prefix.writeline(f"uint32_t grid_0 = static_cast<uint32_t>({grid});")
    prefix.writeline("uint32_t grid_1 = 1;")
    prefix.writeline("uint32_t grid_2 = 1;")


@dataclasses.dataclass
class TritonDeferredNpuCallWrapper(DeferredNpuTritonCallWrapper):
    def generate(self, wrapper):
        # Declare helpers before the deferred kernel wrappers that use them.
        if not getattr(wrapper, "_te_downcast_helpers_emitted", False):
            wrapper.prefix.splice(_DOWNCAST_HELPERS)
            wrapper._te_downcast_helpers_emitted = True
        super().generate(wrapper)

    def _generate_single_kernel_launch(self, prefix, wrapper, kernel_var_name, params):
        downcast_args = params["triton_meta"].get("downcast_args", {})
        mutated = set(params["inductor_meta"].get("mutated_arg_names", ()))
        replacements = {}
        writebacks = []
        for name, arg_type in zip(params["def_args"], self.arg_types):
            if not isinstance(arg_type, torch.dtype):
                continue
            dtype = {
                ("*i64", torch.int64): "int32",
                ("*fp64", torch.float64): "float32",
            }.get((downcast_args.get(name), arg_type))
            if dtype is None:
                continue
            is_mutated = name in mutated or name.startswith("in_out_ptr")
            is_output = is_mutated or name.startswith("out_ptr")
            temporary = f"{name}_downcast"
            preserve_input = str(is_mutated or not is_output).lower()
            prefix.writeline(
                f"auto {temporary} = te_downcast({name}, "
                f"aoti_torch_dtype_{dtype}(), {preserve_input});"
            )
            replacements[name] = temporary
            if is_output:
                writebacks.append((name, temporary))
        cast_params = dict(params)
        for key in ("def_args", "call_args"):
            cast_params[key] = [replacements.get(name, name) for name in params[key]]
        cast_params["triton_meta"] = dict(params["triton_meta"])
        for key in ("signature", "constants"):
            cast_params["triton_meta"][key] = {
                replacements.get(name, name): value
                for name, value in params["triton_meta"][key].items()
            }
        super()._generate_single_kernel_launch(
            prefix, wrapper, kernel_var_name, cast_params
        )
        for name, temporary in writebacks:
            prefix.writeline(f"te_downcast_writeback({name}, {temporary});")

    def generate_grid(self, prefix, inductor_meta, params):
        for name, value in params.get("runtime_blocks", {}).items():
            prefix.writeline(f"int64_t {name} = {value};")
        grid_params = dict(params)
        grid_params["inductor_meta"] = inductor_meta
        _emit_grid(prefix, grid_params)


class TritonCppWrapperNpu(CppWrapperNpu):
    """Reuse the NPU host ABI and override only experimental kernel selection."""

    def __init__(self):
        if not inductor_config.triton.autotune_at_compile_time:
            raise NotImplementedError(
                "triton_experimental C++ wrapper requires "
                "triton.autotune_at_compile_time=True; lazy compilation is not supported"
            )
        super().__init__()

    @staticmethod
    def create(is_subgraph, subgraph_name, parent_wrapper, partition_signatures=None):
        return TritonCppWrapperNpu()

    @classmethod
    def _get_triton_info_kernel_cls(cls):
        from .triton import NPUTritonKernel

        return NPUTritonKernel

    def _generate_kernel_call_helper(self, kernel_name, call_args, **kwargs):
        super()._generate_kernel_call_helper(kernel_name, call_args, **kwargs)
        wrapper = self._triton_call_wrappers.get(f"call_{kernel_name}")
        if isinstance(wrapper, DeferredTritonCallWrapper) and not isinstance(
            wrapper, TritonDeferredNpuCallWrapper
        ):
            self._triton_call_wrappers[wrapper.wrapper_name] = (
                TritonDeferredNpuCallWrapper(
                    wrapper_name=wrapper.wrapper_name,
                    kernel_name=wrapper.kernel_name,
                    kernel_name_to_body=self._kernel_name_to_body,
                    arg_types=wrapper.arg_types,
                    triton_meta=wrapper.triton_meta,
                    inductor_meta=wrapper.inductor_meta,
                    tma_tensor_args=wrapper.tma_tensor_args,
                )
            )
