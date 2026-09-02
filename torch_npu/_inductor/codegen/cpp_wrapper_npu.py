import dataclasses
from contextlib import contextmanager
from itertools import zip_longest
from typing import Any, Optional, Union
from typing_extensions import Self

import sympy

import torch
from torch import dtype as torch_dtype
from torch._inductor import config
from torch._inductor.codecache import (
    CudaKernelParamCache,
    get_cpp_wrapper_cubin_path_name,
)
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.codegen.cpp_utils import cexpr, DEVICE_TO_ATEN, DTYPE_TO_CPP
from torch._inductor.codegen.cpp_wrapper_gpu import (
    cpp_string_literal,
    CppWrapperGpu,
    DeferredTritonCallWrapper,
    UnwrapUnspecArg,
)
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.codegen.wrapper import PythonWrapperCodegen, SymbolicCallArg
from torch._inductor.ir import GraphPartitionSignature
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .. import config as npu_config
from ..runtime.triton_heuristics import GridExprNpu
from ..utils import triton_support_ffts, triton_support_auto_blockify

# follow triton-ascend implement, except torch.bfloat16
# torch.bfloat16 -> "float" will be specially deal in codegen_tensor_item_npu
DTYPE_TO_TA_TYPE = {
    **DTYPE_TO_CPP,
    torch.bool: "int32_t"
}

@dataclasses.dataclass
class DeferredNpuTritonCallWrapper(DeferredTritonCallWrapper):
    """
    When using cpp wrapper, GPU kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as cubin files, so use a deferred generating the final wrapper around
    the triton kernel until right before the prefix is written.
    """

    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]
    triton_meta: dict[str, Any] | None = None
    inductor_meta: dict[str, Any] | None = None
    tma_tensor_args: dict[str, str] = dataclasses.field(default_factory=dict)

    @contextmanager
    def _patch_runtime_block_params(self):
        if self.kernel_name.startswith("multi_kernel_"):
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)
        params = CudaKernelParamCache.get(self.kernel_name)
        if not params:
            raise RuntimeError(
                f"CudaKernelParamCache not populated for {self.kernel_name}"
            )

        original_def_args = params["def_args"]
        original_arg_types = self.arg_types
        inductor_meta = params["inductor_meta"]
        runtime_block_names = tuple(inductor_meta.get("runtime_block_arg_names", ()))

        try:
            if runtime_block_names:
                wrapper_def_args = [
                    name for name in original_def_args if name not in runtime_block_names
                ]
                params["def_args"] = wrapper_def_args

                if (
                    "extra_launcher_args" in inductor_meta
                    and len(wrapper_def_args) > len(original_arg_types)
                ):
                    expected_arg_count = len(original_arg_types) - len(
                        inductor_meta["extra_launcher_args"]
                    )
                    if len(wrapper_def_args) != expected_arg_count:
                        raise RuntimeError(
                            "wrapper_def_args and arg_types do not match for extra_launcher_args: "
                            f"{len(wrapper_def_args)} != {expected_arg_count}"
                        )
                    self.arg_types = original_arg_types + [SymbolicCallArg] * len(
                        inductor_meta["extra_launcher_args"]
                    )

            yield params
        finally:
            params["def_args"] = original_def_args
            self.arg_types = original_arg_types

    def generate(self, wrapper: CppWrapperGpu):
        additional_files = V.graph.wrapper_code.additional_files
        existing_files = OrderedSet(additional_files)
        with self._patch_runtime_block_params() as params:
            # todo: support lazy compile for cpp_wrapper_npu
            with config.patch("triton.autotune_at_compile_time", not V.graph.aot_mode):
                super().generate(wrapper)
            inductor_meta = params["inductor_meta"]
            if inductor_meta.get("group_enabled", False):
                grouped_plan = inductor_meta["grouped_candidate_plan"]
                cubin_paths = tuple(
                    self._grouped_load_meta(grouped_plan, variant_id)[
                        "cubin_path"
                    ]
                    for variant_id in self._grouped_active_variants(grouped_plan)
                )
                default_cubin_path = params[get_cpp_wrapper_cubin_path_name()]
                if (
                    default_cubin_path not in existing_files
                    and default_cubin_path not in cubin_paths
                    and default_cubin_path in additional_files
                ):
                    additional_files.remove(default_cubin_path)
            else:
                cubin_paths = (params[get_cpp_wrapper_cubin_path_name()],)
            for cubin_path in cubin_paths:
                if cubin_path not in additional_files:
                    additional_files.append(cubin_path)

    @staticmethod
    def _grouped_runtime_block_names(
        grouped_plan: dict[str, Any],
    ) -> tuple[str, ...]:
        return tuple(grouped_plan.get("runtime_block_append_order", ()))

    @staticmethod
    def _grouped_active_variants(
        grouped_plan: dict[str, Any],
    ) -> tuple[str, ...]:
        best_by_group = grouped_plan.get("best_by_group", {})
        if not best_by_group:
            raise RuntimeError(
                "grouped cpp wrapper expects best_by_group to be populated"
            )
        selected_variant_ids = OrderedSet(
            selected["variant_id"] for selected in best_by_group.values()
        )
        active_variants = tuple(
            variant_id
            for variant_id in grouped_plan.get("variant_order", ())
            if variant_id in selected_variant_ids
        )
        if not active_variants:
            raise RuntimeError(
                "grouped cpp wrapper has no active compiled variants"
            )
        return active_variants

    @staticmethod
    def _grouped_load_meta(
        grouped_plan: dict[str, Any], variant_id: str
    ) -> dict[str, Any]:
        load_meta = dict(
            grouped_plan["variants"][variant_id].get("load_meta", {})
        )
        if not load_meta:
            raise RuntimeError(
                f"grouped cpp wrapper expects load_meta for variant {variant_id}"
            )
        return load_meta

    def _generate_grouped_feature_inputs(
        self,
        prefix: IndentedBuffer,
        grouped_plan: dict[str, Any],
        def_args: list[str],
    ) -> None:
        feature_arg_indices = tuple(
            grouped_plan.get("feature_arg_indices", ())
        )
        feature_sources = tuple(grouped_plan.get("feature_sources", ()))
        if len(feature_arg_indices) != len(feature_sources):
            raise RuntimeError(
                "grouped cpp wrapper feature inputs and sources do not match"
            )
        for feature_idx, (arg_indices, feature_source) in enumerate(
            zip(feature_arg_indices, feature_sources)
        ):
            arg_names = [def_args[arg_index] for arg_index in arg_indices]
            source = feature_source.get("source")
            if source in ("outer_product", "reduction_product"):
                feature_expr = " * ".join(arg_names)
            elif len(arg_names) == 1:
                feature_expr = arg_names[0]
            else:
                raise RuntimeError(
                    f"grouped cpp wrapper feature {source} expects one axis"
                )
            prefix.writeline(
                f"int64_t grouped_feature_{feature_idx} = {feature_expr};"
            )

    def _generate_grouped_group_id(
        self, prefix: IndentedBuffer, grouped_plan: dict[str, Any]
    ) -> None:
        group_features = tuple(grouped_plan.get("group_features", ()))
        prefix.writeline("int64_t grouped_group_id = 0;")
        prefix.writeline("int64_t grouped_group_stride = 1;")
        for feature_idx, feature_spec in enumerate(group_features):
            buckets = tuple(feature_spec.get("buckets", ()))
            prefix.writeline(f"int64_t grouped_bucket_{feature_idx} = 0;")
            for bucket_idx, upper_bound in enumerate(buckets):
                keyword = "if" if bucket_idx == 0 else "else if"
                prefix.writeline(
                    f"{keyword} (grouped_feature_{feature_idx} <= "
                    f"{int(upper_bound)}) grouped_bucket_{feature_idx} = "
                    f"{bucket_idx};"
                )
            if buckets:
                prefix.writeline(
                    f"else grouped_bucket_{feature_idx} = {len(buckets)};"
                )
            prefix.writeline(
                f"grouped_group_id += grouped_bucket_{feature_idx} * "
                "grouped_group_stride;"
            )
            prefix.writeline(
                f"grouped_group_stride *= {len(buckets) + 1};"
            )
        group_id_count = int(grouped_plan.get("group_id_count", 0))
        if group_id_count:
            prefix.writeline(
                f"if (grouped_group_id >= {group_id_count}) "
                'throw std::runtime_error("grouped cpp wrapper resolved group '
                'id out of range");'
            )

    def _generate_grouped_selection(
        self,
        prefix: IndentedBuffer,
        grouped_plan: dict[str, Any],
        def_args: list[str],
        active_variants: tuple[str, ...],
    ) -> None:
        runtime_block_names = self._grouped_runtime_block_names(grouped_plan)
        axis_arg_indices = dict(grouped_plan.get("axis_arg_indices", {}))
        best_by_group = grouped_plan["best_by_group"]
        prefix.writeline("int64_t grouped_variant_index = -1;")
        for block_name in runtime_block_names:
            prefix.writeline(f"int64_t {block_name} = 0;")
        if runtime_block_names:
            prefix.splice(
                """
                auto resolve_grouped_runtime_block = [](
                    int64_t axis_numel,
                    int64_t expected_grid,
                    int64_t block_sub
                ) -> int64_t {
                    if (axis_numel <= 0) return 1;
                    auto ceildiv = [](int64_t value, int64_t divisor) {
                        return (value + divisor - 1) / divisor;
                    };
                    int64_t total_subblocks = ceildiv(axis_numel, block_sub);
                    int64_t program_subblocks = ceildiv(
                        total_subblocks, expected_grid
                    );
                    int64_t effective_grid = ceildiv(
                        total_subblocks, program_subblocks
                    );
                    return ceildiv(axis_numel, effective_grid);
                };
                """
            )
        prefix.writeline("switch (grouped_group_id) {")
        with prefix.indent():
            for group_id in grouped_plan.get("reachable_group_ids", ()):
                selected = best_by_group.get(str(group_id))
                if selected is None:
                    raise RuntimeError(
                        "grouped cpp wrapper is missing winner for reachable "
                        f"group {group_id}"
                    )
                variant_id = selected["variant_id"]
                policy = grouped_plan["policies"][selected["policy_id"]]
                prefix.writeline(f"case {group_id}: {{")
                with prefix.indent():
                    prefix.writeline(
                        f"grouped_variant_index = "
                        f"{active_variants.index(variant_id)};"
                    )
                    assigned_blocks = OrderedSet()
                    for block_name, block_value in policy.get(
                        "static_blocks", ()
                    ):
                        prefix.writeline(
                            f"{block_name} = {int(block_value)};"
                        )
                        assigned_blocks.add(block_name)
                    for block_name, rule_items in policy.get(
                        "runtime_block_rules", ()
                    ):
                        rule = dict(rule_items)
                        if rule.get("op") != "ceildiv":
                            raise RuntimeError(
                                "grouped cpp wrapper only supports ceildiv "
                                "runtime block rules"
                            )
                        axis_name = rule["axis_name"]
                        if axis_name not in axis_arg_indices:
                            raise RuntimeError(
                                "grouped cpp wrapper is missing axis argument "
                                f"for {axis_name}"
                            )
                        axis_arg = def_args[axis_arg_indices[axis_name]]
                        prefix.writeline(
                            f"{block_name} = resolve_grouped_runtime_block("
                            f"{axis_arg}, {int(policy['grid_target'])}, "
                            f"{int(rule['block_sub'])});"
                        )
                        assigned_blocks.add(block_name)
                    missing_blocks = [
                        name
                        for name in runtime_block_names
                        if name not in assigned_blocks
                    ]
                    if missing_blocks:
                        raise RuntimeError(
                            "grouped cpp wrapper policy is missing runtime "
                            f"blocks {missing_blocks}"
                        )
                    prefix.writeline("break;")
                prefix.writeline("}")
            prefix.writeline("default:")
            with prefix.indent():
                prefix.writeline(
                    'throw std::runtime_error("grouped cpp wrapper resolved '
                    'an unavailable group");'
                )
        prefix.writeline("}")

    def _generate_grouped_variant_launch(
        self,
        prefix: IndentedBuffer,
        wrapper: CppWrapperGpu,
        params: dict[str, Any],
        grouped_plan: dict[str, Any],
        active_variants: tuple[str, ...],
    ) -> None:
        prefix.writeline("switch (grouped_variant_index) {")
        with prefix.indent():
            for variant_index, variant_id in enumerate(active_variants):
                load_meta = self._grouped_load_meta(grouped_plan, variant_id)
                prefix.writeline(f"case {variant_index}: {{")
                with prefix.indent():
                    kernel_var_name = f"grouped_kernel_{variant_id}"
                    variant_params = {
                        **params,
                        **load_meta,
                    }
                    self._generate_single_kernel_load(
                        prefix, kernel_var_name, variant_params
                    )
                    self._generate_single_kernel_launch(
                        prefix,
                        wrapper,
                        kernel_var_name,
                        variant_params,
                    )
                    prefix.writeline("break;")
                prefix.writeline("}")
            prefix.writeline("default:")
            with prefix.indent():
                prefix.writeline(
                    'throw std::runtime_error("grouped cpp wrapper could not '
                    'launch selected variant");'
                )
        prefix.writeline("}")

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ):
        if inductor_meta.get("group_enabled", False):
            grouped_plan = inductor_meta["grouped_candidate_plan"]
            active_variants = self._grouped_active_variants(grouped_plan)
            self._generate_grouped_feature_inputs(
                prefix, grouped_plan, params["def_args"]
            )
            self._generate_grouped_group_id(prefix, grouped_plan)
            self._generate_grouped_selection(
                prefix,
                grouped_plan,
                params["def_args"],
                active_variants,
            )
        numels = [arg for arg in params["def_args"] if "_numel" in arg]
        if not inductor_meta.get("group_enabled", False):
            for block_name, block_value in dict(
                params.get("runtime_blocks", {})
            ).items():
                prefix.writeline(f"int64_t {block_name} = {block_value};")
        grid = GridExprNpu.from_meta_and_set_numel(
            inductor_meta, params["config"], numels, "cpp"
        )
        for line in grid.prefix:
            prefix.writeline(line)
        prefix.splice(
            f"""\
            uint32_t grid_0 = {grid.x_grid};
            uint32_t grid_1 = {grid.y_grid};
            uint32_t grid_2 = {grid.z_grid};
            """
        )
        prefix.writeline("if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;")

    def generate_load_kernel(self, prefix, kernel_var_name, params):
        inductor_meta = params["inductor_meta"]
        if inductor_meta.get("group_enabled", False):
            grouped_plan = inductor_meta["grouped_candidate_plan"]
            for variant_id in self._grouped_active_variants(grouped_plan):
                prefix.writeline(
                    f"static void* grouped_kernel_{variant_id} = nullptr;"
                )
            return
        self._generate_single_kernel_load(prefix, kernel_var_name, params)

    def _generate_single_kernel_load(self, prefix, kernel_var_name, params):
        prefix.writeline(f"if ({kernel_var_name} == nullptr) {{")
        with prefix.indent():
            embed_kernel_args = [f"__{params['inductor_meta']['kernel_name']}_start"]

            if V.graph.aot_mode and config.aot_inductor.embed_kernel_binary:
                load_kernel_args = [
                    *embed_kernel_args,
                    cpp_string_literal(params["mangled_name"]),
                    # add mix_mode into kernel load param
                    cpp_string_literal(params["mix_mode"]),
                    str(params["shared_mem"]),
                ]
            else:
                load_kernel_args = [
                    cpp_string_literal(params[get_cpp_wrapper_cubin_path_name()]),
                    cpp_string_literal(params["mangled_name"]),
                    # add mix_mode into kernel load param
                    cpp_string_literal(params["mix_mode"]),
                    str(params["shared_mem"]),
                    "cubin_dir_",
                ]

            prefix.writeline(
                f"{kernel_var_name} = loadKernel({', '.join(load_kernel_args)}); "
            )
        prefix.writeline("}")

    def generate_launch_kernel(self, prefix, wrapper, kernel_var_name, params):
        inductor_meta = params["inductor_meta"]
        if inductor_meta.get("group_enabled", False):
            grouped_plan = inductor_meta["grouped_candidate_plan"]
            self._generate_grouped_variant_launch(
                prefix,
                wrapper,
                params,
                grouped_plan,
                self._grouped_active_variants(grouped_plan),
            )
            return
        self._generate_single_kernel_launch(
            prefix, wrapper, kernel_var_name, params
        )

    def _generate_single_kernel_launch(
        self, prefix, wrapper, kernel_var_name, params
    ):
        triton_meta = params["triton_meta"]
        arg_type_lookup = dict(zip(params["def_args"], self.arg_types))
        for block_name in params["inductor_meta"].get("runtime_block_arg_names", ()):
            arg_type_lookup.setdefault(block_name, SymbolicCallArg)
        # difference between Python and C++ wrapper: C++ wrapper strips out equal_to_1 constants
        call_args = [
            name for name in params["call_args"] if name not in triton_meta["constants"]
        ]
        arg_types = [arg_type_lookup[name] for name in call_args]
        arg_signatures = [triton_meta["signature"][name] for name in call_args]
        is_pure_simt = npu_config.is_ascend950 and params["is_pure_simt"]
        enable_simt = npu_config.is_ascend950 and (
            "simt" in params["parallel_mode"] or params["is_pure_simt"]
        )
        enable_auto_blockify = not params.get(
            "has_auto_blockify_blacklist_op", False
        ) and triton_support_auto_blockify()
        args_decl = wrapper.generate_args_decl(
            prefix,
            call_args,
            arg_types,
            arg_signatures,
            is_triton_kernel=True,
            is_pure_simt=is_pure_simt,
            kernel_params=params,
        )
        prefix.splice(f"""
        auto launch_call = [=]() {{
        {args_decl}
        {wrapper.generate_launch_preparation(kernel_var_name, params, enable_simt, enable_auto_blockify)}
        }};
        """)
        prefix.writeline(
            r"launchKernel({}, {});".format("launch_call", f'"{kernel_var_name}"')
        )
        # todo: add support for enable_kernel_profile


class CppWrapperNpu(CppWrapperGpu):
    """
    Generates cpp wrapper for running on NPU and calls CUDA kernels
    """

    def __init__(self) -> None:
        self.device = "npu"
        self.device_codegen = get_device_op_overrides(self.device)
        super().__init__()

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[GraphPartitionSignature] = None,
    ):
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperNpu()

    def write_header(self):
        self.header.splice(f"""
            #include <torch_npu/csrc/inductor/aoti_runtime/model.h>
            #include <torch_npu/csrc/inductor/aoti_runtime/device_utils.h>
            #include <torch_npu/csrc/inductor/aoti_runtime/utils_npu.h>
            #include <torch_npu/csrc/inductor/aoti_torch/generated/c_shim_{self.device}.h>
        """)
        super().write_header()
        self.header = self.header.map(
            lambda line: line.replace(
                "<torch/csrc/inductor/aoti_include/common.h>",
                "<torch_npu/csrc/inductor/aoti_include/common.h>") if isinstance(line, str) else line)

    def generate_node_numel_expr(self, kernel_name: str, node, numel_expr):
        expr = f"{kernel_name}_{node.name}_numel"

        if (expr, V.graph) not in self.kernel_numel_expr:
            # declare expr once in each graph (scope)
            self.kernel_numel_expr.add((expr, V.graph))
            self.writeline(f"int64_t {expr} = {cexpr(numel_expr)};")
        else:
            self.writeline(f"{expr} = {cexpr(numel_expr)};")
        return SymbolicCallArg(expr, numel_expr)

    @classmethod
    def _get_triton_info_kernel_cls(cls):
        from torch_npu._inductor.codegen.triton import NPUTritonKernel

        return NPUTritonKernel

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        inductor_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ):
        if (
            not triton
            and not V.graph.aot_mode
            and kernel_name in self.initialized_kernels
        ):
            # CATLASS kernel in JIT cpp_wrapper mode.
            #
            # In the community CUTLASS flow, AOT mode links the kernel .o into
            # model.so and calls it through a function pointer in the
            # AOTInductorModelKernels class (kernels.<name>(...)).  In JIT mode
            # the community does not support CUTLASS cpp_wrapper.  We extend it
            # by loading the standalone .so via dlopen/dlsym at first call,
            # which is the CATLASS analog of Triton's loadKernel/launchKernel
            # pattern.  The load_* helper and function pointer are emitted in
            # finalize_prefix().
            device = device or V.graph.get_current_device_or_throw()
            stream = self.write_get_raw_stream(device.index, graph_name)

            # Build the casted argument list (same logic as CppWrapperGpu).
            casted = []
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if isinstance(arg_type, str) and arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                casted.append(f"({arg_type}){cexpr(new_arg)}")
            call_args_str = ", ".join(casted)

            # Lazy-load the .so on first call, then invoke through the
            # function pointer.
            self.writeline(f"load_{kernel_name}();")
            # Synchronize the stream before launching the catlass kernel.
            # Catlass kernels read input tensors (e.g., offsets/group_list
            # produced by aten ops like cumsum) directly from device memory.
            # Aten ops in cpp_wrapper use the implicit "current stream" which
            # may differ from the explicit stream used by catlass, or the
            # AscendC <<<>>> launch may not fully respect stream ordering with
            # preceding aten ops.  This sync ensures all prior operations
            # (including aten ops that produce catlass inputs) have completed
            # before the catlass kernel reads them.
            self.writeline(f"aclrtSynchronizeStream({stream});")
            self.writeline(f"{kernel_name}({call_args_str}, {stream});")
            return

        super()._generate_kernel_call_helper(
            kernel_name,
            call_args,
            device=device,
            triton=triton,
            arg_types=arg_types,
            raw_keys=raw_keys,
            raw_args=raw_args,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            graph_name=graph_name,
            original_fxnode_name=original_fxnode_name,
            current_stream_idx=current_stream_idx,
        )
        wrapper_name = f"call_{kernel_name}"
        if wrapper_name in self._triton_call_wrappers:
            # trans DeferredTritonCallWrapper to DeferredNpuTritonCallWrapper
            wrapper = self._triton_call_wrappers[wrapper_name]
            npu_wrapper = DeferredNpuTritonCallWrapper(
                wrapper_name=wrapper.wrapper_name,
                kernel_name=wrapper.kernel_name,
                kernel_name_to_body=self._kernel_name_to_body,
                arg_types=wrapper.arg_types,
                triton_meta=wrapper.triton_meta,
                inductor_meta=wrapper.inductor_meta,
                tma_tensor_args=wrapper.tma_tensor_args,
            )
            self._triton_call_wrappers[wrapper_name] = npu_wrapper

    @staticmethod
    def get_device_include_path_jit(device: str) -> str:
        return f"#include <torch_npu/csrc/inductor/cpp_wrapper/{device}.h>"

    @staticmethod
    def get_device_include_path_aot(device: str) -> str:
        return f"#include <torch_npu/csrc/inductor/aoti_include/{device}.h>"

    def add_device_include(self, device: str) -> None:
        # add acl, rt and torch_npu include
        self.header.splice("""
            #include <fstream>
            #include <acl/acl.h>
            #include <acl/acl_rt.h>
            #include <runtime/runtime/rt.h>
            #include <torch_npu/csrc/core/npu/NPUStream.h>
            #include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
            #include <torch_npu/csrc/framework/OpCommand.h>
        """)
        super().add_device_include(device)

    def _write_aoti_interface_header(self):
        super()._write_aoti_interface_header()
        self.header = self.header.map(
            lambda line: line.replace(
                "<torch/csrc/inductor/aoti_runtime/model_container.h>",
                "<torch_npu/csrc/inductor/aoti_runtime/model_container.h>") if isinstance(line, str) else line)
        self.header = self.header.map(
            lambda line: line.replace(
                "cuda",
                "npu") if isinstance(line, str) else line)

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperNpu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def finalize_prefix(self):
        """Generate C++ declarations for CATLASS kernels before the parent
        finalises the prefix.

        AOT mode:
            The parent ``codegen_model_kernels()`` (called via
            ``CppWrapperCpu.finalize_prefix``) emits ``extern "C"`` declarations
            and function-pointer bindings for every kernel in
            ``initialized_kernels``, and the CATLASS .o files are linked into
            model.so via ``CATLASSCodeCache.aot_kernels_o`` — exactly the same
            flow as the community CUTLASS backend.

        JIT cpp_wrapper mode:
            The community does not support CUTLASS in JIT cpp_wrapper.  We
            extend it by emitting a function-pointer typedef, a static pointer
            initialised to ``nullptr``, and a ``load_*`` helper that uses
            ``dlopen``/``dlsym`` on first call.  The .so path is computed from
            ``src_to_kernel`` + ``_catlass_kernel_is_mix`` (populated in
            ``CATLASSScheduling.define_kernel``) so that no redundant
            compilation is triggered.
        """
        old_prefix = self.prefix

        # Generate C++ declarations for CATLASS kernels (JIT mode only).
        self.prefix = IndentedBuffer()
        if not V.graph.aot_mode and self.initialized_kernels:
            import textwrap as _textwrap
            from torch_npu._inductor.codecache import CATLASSCodeCache

            # Build reverse lookup: kernel_name -> original src_code
            src_to_kernel = V.graph.wrapper_code.src_to_kernel
            kernel_to_src = {v: k for k, v in src_to_kernel.items()}
            is_mix_map = getattr(V.graph.wrapper_code, "_catlass_kernel_is_mix", {})

            # Ensure <dlfcn.h> is available for dlopen/dlsym.
            self.prefix.writeline("#include <dlfcn.h>")
            self.prefix.writeline("")

            # CATLASS kernels use catlass-native type names (bfloat16_t, half)
            # in their signatures.  Provide aliases so the C++ wrapper code
            # (which uses the at:: namespace) can reference them.
            self.prefix.writeline("// CATLASS-native type aliases for cpp_wrapper")
            self.prefix.writeline("using bfloat16_t = at::BFloat16;")
            self.prefix.writeline("using half = at::Half;")
            self.prefix.writeline("")

            for kernel_name, kernel in self.initialized_kernels.items():
                signature = kernel.get_signature()

                # Build a typedef by replacing the function name with
                # "(*typedef_name)".
                typedef_sig = signature.replace(
                    kernel_name, f"(*{kernel_name}_t)"
                )
                self.prefix.writeline(f"typedef {typedef_sig};")
                self.prefix.writeline(
                    f"static {kernel_name}_t {kernel_name} = nullptr;"
                )
                self.prefix.writeline(
                    f"static void* {kernel_name}_so_handle = nullptr;"
                )
                self.prefix.writeline("")

                # Compute the .so path from the source code that
                # async_compile.catlass() will compile.  We must use the
                # *exact* source string: splice(src_code, strip=True)
                # produces "\n" + textwrap.dedent(src_code).strip() + "\n".
                # CATLASSCodeCache.write() only writes the .cpp source file
                # (no compilation), so this is cheap and the resulting path
                # matches the .so that async_compile.catlass() produces.
                original_src = kernel_to_src.get(kernel_name, "")
                compiled_src = original_src.replace("KERNEL_NAME", kernel_name)
                autotune_src = "\n" + _textwrap.dedent(compiled_src).strip() + "\n"
                is_mix = is_mix_map.get(kernel_name, False)
                _key, _input_path = CATLASSCodeCache.write(
                    autotune_src, "so", is_mix
                )
                so_path = (
                    _input_path[: -len(CATLASSCodeCache._SOURCE_CODE_SUFFIX)] + "so"
                )

                self.prefix.splice(f"""
static inline void load_{kernel_name}() {{
    if ({kernel_name} == nullptr) {{
        const char* so_path = "{so_path}";
        {kernel_name}_so_handle = dlopen(so_path, RTLD_NOW);
        if ({kernel_name}_so_handle == nullptr) {{
            throw std::runtime_error(std::string("Failed to dlopen catlass kernel: ") + so_path + " error: " + dlerror());
        }}
        {kernel_name} = reinterpret_cast<{kernel_name}_t>(dlsym({kernel_name}_so_handle, "{kernel_name}"));
        if ({kernel_name} == nullptr) {{
            throw std::runtime_error(std::string("Failed to dlsym catlass kernel '{kernel_name}' in: ")
                + so_path + " error: " + dlerror());
        }}
    }}
}}
""")
                self.prefix.writeline("")

        catlass_prefix = self.prefix

        # Now let the parent (CppWrapperGpu -> CppWrapperCpu) do its work.
        # In AOT mode, this calls codegen_model_kernels() which emits
        # extern "C" declarations and function-pointer bindings for
        # initialized_kernels — the same flow as community CUTLASS.
        self.prefix = IndentedBuffer()
        super().finalize_prefix()

        # Prepend our CATLASS declarations.
        self.prefix.splice(catlass_prefix)
        self.prefix.writeline("\n")
        self.prefix.splice(old_prefix)

    def codegen_initialized_kernel_decls(self):
        # In AOT mode, CATLASS .o files are linked into model.so, so the
        # parent's `extern "C"` declarations are required — same as the
        # community CUTLASS flow.
        # In JIT cpp_wrapper mode, CATLASS kernels are loaded dynamically
        # via dlopen/dlsym through function pointers emitted in
        # finalize_prefix().  An `extern "C"` function declaration here
        # would conflict with the `static <name>_t <name> = nullptr;`
        # pointer variable (C++ forbids a function and a variable sharing
        # the same identifier), so we skip it.  In JIT mode
        # initialized_kernels only contains CATLASS kernels on NPU.
        if not V.graph.aot_mode:
            return
        super().codegen_initialized_kernel_decls()

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_TA_TYPE[dtype]} {scalar_tmp};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
            struct_data = f"float {scalar} __attribute__((aligned(4)));"
            arg_data = f"static_cast<float>({scalar})"
        else:
            writer.writeline(f"{DTYPE_TO_TA_TYPE[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )
            struct_data = f"{DTYPE_TO_TA_TYPE[dtype]} {scalar} __attribute__((aligned(sizeof({DTYPE_TO_TA_TYPE[dtype]} ))));"
            arg_data = f"static_cast<{DTYPE_TO_TA_TYPE[dtype]}>({scalar})"

        return struct_data, arg_data

    def codegen_device(self, device):
        if device.type not in DEVICE_TO_ATEN:
            raise RuntimeError(device.type + "not found in DEVICE_TO_ATEN")
        device_str = DEVICE_TO_ATEN[device.type][5:].lower()  # remove "at::k"
        if device_str == "privateuse1":
            device_str = "npu"
        self.used_cached_devices.add(device_str)
        return f"cached_torch_device_type_{device_str}, {device.index if device.index else 0}"

    def generate_args_decl(
        self,
        code: Union[IndentedBuffer, Self],
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel=True,
        is_pure_simt=False,
        kernel_params: Optional[dict[str, Any]] = None,
    ):
        """
        Generates any declarations of args to pass into a kernel call, and then returns the arg names.

        In more detail:
        * declarations: e.g. this function has a side effect of generating lines like `auto var_0 = ...;`
        * returns: a string with the list of args, e.g. "var_0, var_1"

        call_args: list of call arguments
        arg_types: list of argument types
        arg_signatures: list with signatures of all the args
        is_triton_kernel: whether these are passed into a triton kernel or not. In particular,
                          calls to triton kernels will have an additional global scratch space
                          arg injected at the front of the arg list.
        """

        # Add more cases for other types as needed
        signature2dtype = {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint32_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
        }

        struct_def_body = ""
        struct_arg_body = ""

        target_support_ffts = triton_support_ffts()
        kernel_params = kernel_params or {}
        lock_num = int(kernel_params.get("lock_num", 0) or 0)
        lock_init_val = int(kernel_params.get("lock_init_val", 0) or 0)
        workspace_size = int(kernel_params.get("workspace_size", 0) or 0)

        def process_args(arg, arg_type, arg_signature=None):
            var_name = f"var_{next(self.arg_var_id)}"
            # ignore nvTmaDesc, as host-side TMA descriptors need
            # to be passed to the compiled Triton kernel by value
            if isinstance(arg_type, UnwrapUnspecArg) and arg_signature != "nvTmaDesc":
                struct_data, arg_data = self.codegen_tensor_item(
                    arg_type.dtype,
                    arg,
                    var_name,
                    indented_buffer=code,
                )
            elif isinstance(arg_type, torch_dtype) and arg_signature != "nvTmaDesc":
                device_ptr_type = self.device_codegen.cpp_device_ptr()
                code.writeline(
                    maybe_hipify_code_wrapper(
                        f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                    )
                )
                struct_data = f"void* {var_name} __attribute__((aligned(8)));"
                arg_data = f"static_cast<void*>({var_name})"
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int32_t {var_name} = {cexpr(arg)};")
                struct_data = f"int32_t {var_name} __attribute__((aligned(4)));"
                arg_data = f"static_cast<int32_t>({var_name})"
            elif arg_type in (sympy.Float, float):
                code.writeline(f"float {var_name} = {cexpr(arg)};")
                struct_data = f"float {var_name} __attribute__((aligned(4)));"
                arg_data = f"static_cast<float>({var_name})"
            # For symbolic call arguments, examine the arg signatures from triton meta
            # to explicitly cast to the right type
            # Reason: `auto` can infer unexpected type against kernel input signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in signature2dtype
            ):
                arg_dtype = signature2dtype[arg_signature]
                code.writeline(f"{arg_dtype} {var_name} = {cexpr(arg)};")
                struct_data = f"{arg_dtype} {var_name} __attribute__((aligned(sizeof({arg_dtype}))));"
                arg_data = f"static_cast<{arg_dtype}>({var_name})"
            else:
                raise TypeError("Infer arg_type to cpp failed!")
            nonlocal struct_def_body
            nonlocal struct_arg_body
            struct_def_body += struct_data + " "
            struct_arg_body += arg_data + ", "

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        ffts_str = """
            void* ffts_addr = NULL;
            ret = aclrtGetHardwareSyncAddr(&ffts_addr);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error(std::string("aclrtGetHardwareSyncAddr failed, 0x") + std::to_string(ret));
            }
            """

        workspace_str = ""
        if not is_pure_simt and workspace_size > 0:
            workspace_str = f"""
            uint64_t workspace_size = static_cast<uint64_t>({workspace_size})
                * grid_0 * grid_1 * grid_2;
            auto workspace_tensor = at_npu::native::allocate_workspace(
                workspace_size, stream_);
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
            """

        sync_block_lock_str = ""
        if not is_pure_simt and lock_num > 0:
            sync_block_lock_str = f"""
            uint64_t sync_block_lock_size = static_cast<uint64_t>({lock_num})
                * sizeof(int64_t);
            auto sync_block_lock_tensor = at_npu::native::allocate_workspace(
                sync_block_lock_size, stream_);
            sync_block_lock = const_cast<void *>(
                sync_block_lock_tensor.storage().data());
            std::vector<int64_t> sync_block_lock_init(
                {lock_num}, static_cast<int64_t>({lock_init_val}));
            ret = aclrtMemcpy(
                sync_block_lock,
                sync_block_lock_size,
                sync_block_lock_init.data(),
                sync_block_lock_size,
                ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {{
                throw std::runtime_error(
                    std::string("initialize Triton sync block lock failed, 0x")
                    + std::to_string(ret));
            }}
            """

        args_str = f"""
            aclError ret;
            {ffts_str if target_support_ffts else ""}
            {"void* workspace_addr = NULL;" if not is_pure_simt else ""}
            {"void* sync_block_lock = NULL;" if not is_pure_simt else ""}
            {workspace_str}
            {sync_block_lock_str}
            struct __attribute__((packed)) {{
                {"void* ffts_addr __attribute__((aligned(8)));" if target_support_ffts else ""}
                {"void* sync_block_lock __attribute__((aligned(8)));" if not is_pure_simt else ""}
                {"void* workspace_addr __attribute__((aligned(8)));" if not is_pure_simt else ""}
                {struct_def_body}
                int32_t grid_0 __attribute__((aligned(4)));
                int32_t grid_1 __attribute__((aligned(4)));
                int32_t grid_2 __attribute__((aligned(4)));
            }} kernel_args = {{
                {"static_cast<void*>(ffts_addr)," if target_support_ffts else ""}
                {"static_cast<void*>(sync_block_lock)," if not is_pure_simt else ""}
                {"static_cast<void*>(workspace_addr)," if not is_pure_simt else ""}
                {struct_arg_body}
                static_cast<int32_t>(grid_0),
                static_cast<int32_t>(grid_1),
                static_cast<int32_t>(grid_2)
        }};"""

        return args_str

    def generate_launch_preparation(self, kernel_var_name, params, enable_simt, enable_auto_blockify):
        if enable_simt:
            shared_mem_dynamic_size = params["shared_mem_dynamic_size"]
            cpp_kernel_launch = f"""
            aclrtLaunchKernelAttr attrInfo = {{}};
            attrInfo.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
            aclrtLaunchKernelAttrValue value = {{}};
            value.localMemorySize = {shared_mem_dynamic_size};
            attrInfo.value = value;

            aclrtLaunchKernelCfg cfgCfgInfo = {{}};
            cfgCfgInfo.attrs = &attrInfo;
            cfgCfgInfo.numAttrs = 1;
            void* args = static_cast<void*>(&kernel_args);
            auto args_size = sizeof(kernel_args);
            ret = aclrtLaunchKernelWithHostArgs({kernel_var_name}, block_num, stream_, &cfgCfgInfo, args, args_size, nullptr, 0);
            """
        else:
            cpp_kernel_launch = f"""
            void* args = static_cast<void*>(&kernel_args);
            auto args_size = sizeof(kernel_args);
            ret = aclrtLaunchKernelWithHostArgs({kernel_var_name}, block_num, stream_, nullptr, args, args_size, nullptr, 0);
            """

        launch_str = f"""
            uint32_t block_num = grid_0 * grid_1 * grid_2;
            {f'block_num = std::min(block_num, (uint32_t){str(npu_config.num_vector_core)});' if enable_auto_blockify else ''}
            {cpp_kernel_launch}
            if (ret != ACL_SUCCESS) return ret;
            return ret;"""
        return launch_str
