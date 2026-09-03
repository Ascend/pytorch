import builtins
import contextlib
import dataclasses
import functools
import inspect
import itertools
import logging
import math
import os
import time
from concurrent.futures import as_completed, ThreadPoolExecutor, Future
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import counters, dynamo_timed, identity
from torch._inductor import config, ir
from torch._inductor.ir import ChoiceCaller
from torch._inductor.utils import restore_stdout_stderr, sympy_product, unique, Placeholder
from torch._inductor.virtualized import V
from torch._inductor.codegen.triton import (
    texpr,
    TritonScheduling,
)
from torch._inductor.codecache import PyCodeCache
from torch._inductor.autotune_process import (
    TensorMeta,
    TritonBenchmarkRequest,
    TritonCPUBenchmarkRequest,
    TritonGPUBenchmarkRequest,
)
from torch._inductor.select_algorithm import (
    TritonTemplate,
    TritonTemplateKernel,
    VERIFY,
    DEBUG,
    get_mm_log_filename,
    append_to_log,
    get_num_workers,
    NoValidChoicesError,
    create_inputs_key,
    create_precompile_key,
    ExternKernelCaller,
    TritonTemplateCaller,
    AutotuneArgs,
)
from torch._inductor.codegen.common import IndentedBuffer, RemovedArg
from torch._inductor.exc import CppCompileError
from torch.utils._ordered_set import OrderedSet

from ..profiler import tensorboard_trace_handler
from .codegen.triton import NPUTritonKernel
from . import config as npu_config


log = logging.getLogger("torch._inductor")


@dataclasses.dataclass(frozen=True)
class NPUTemplateCompileOption:
    """Compile-only options owned by one NPU Triton template."""

    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    def apply(self, meta: dict[str, Any]) -> None:
        for key, value in self.options.items():
            meta.setdefault(key, value)


def _gen_npu_template_triton_imports() -> str:
    return NPUTritonKernel.gen_common_triton_imports()


def _add_npu_template_meta_to_inductor_meta(
    inductor_meta: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    for key in ("ENABLE_COMPILE_HINT", "BLOCK_M", "BLOCK_N"):
        if key in meta:
            inductor_meta[key] = meta[key]


def _add_npu_template_compile_options_to_triton_meta(
    triton_meta: dict[str, Any],
    meta: dict[str, Any],
    compile_option_keys: frozenset[str],
) -> None:
    compile_options = {
        key: meta[key]
        for key in compile_option_keys
        if key in meta
    }
    if compile_options:
        triton_meta.setdefault("npu_compile_options", {}).update(compile_options)


def _extract_choice_debug_config(choice: ChoiceCaller) -> Optional[Any]:
    for attr in (
        "_sparse_mask_report_config",
        "_flex_attention_report_config",
        "log_info",
        "config",
    ):
        if hasattr(choice, attr):
            value = getattr(choice, attr)
            if value is not None:
                return value
    return None


def _format_choice_debug_label(choice: ChoiceCaller) -> str:
    choice_name = getattr(choice, "name", type(choice).__name__)
    config_info = _extract_choice_debug_config(choice)
    if config_info is not None:
        return f"{choice_name} config={config_info}"
    return str(choice)


def _tiling_sort_key(choice: ChoiceCaller):
    config_info = _extract_choice_debug_config(choice)
    if isinstance(config_info, dict):
        block_m = config_info.get("MASK_BLOCK_M", config_info.get("BLOCK_M"))
        block_n = config_info.get("MASK_BLOCK_N", config_info.get("BLOCK_N"))
        block_m2 = config_info.get("BLOCK_M2")
        block_n2 = config_info.get("BLOCK_N2")
        if block_m is not None or block_n is not None:
            block_m = int(block_m) if block_m is not None else 1
            block_n = int(block_n) if block_n is not None else 1
            if block_m2 is None or block_n2 is None:
                return (block_m * block_n, block_m, block_n)
            block_m2 = int(block_m2)
            block_n2 = int(block_n2)
            return (
                block_m * block_n + block_m2 * block_n2,
                block_m * block_n,
                block_m2 * block_n2,
                block_m,
                block_n,
                block_m2,
                block_n2,
            )
    return None


def _select_first_usable_choice_in_order(
    choices,
    timings,
    successful_precompile_choice_hashes,
):
    for choice in choices:
        if choice in timings:
            return choice
    for choice in choices:
        if choice.hash_key() in successful_precompile_choice_hashes:
            return choice
    tiling_choices = [
        (tiling_key, choice)
        for choice in choices
        if (tiling_key := _tiling_sort_key(choice)) is not None
    ]
    if tiling_choices:
        return max(tiling_choices, key=lambda item: item[0])[1]
    return None


class NPUCompileError(CppCompileError):
    pass


class _NPUTritonBenchmarkRequestMixin:
    kernel_has_output_arg = True

    def do_bench(
        self,
        fn: Callable[[], None],
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> float:
        from torch._inductor.runtime.benchmarking import benchmarker

        device_idx_set = OrderedSet(
            tensor.device.index
            for tensor in (*input_tensors, out)
            if isinstance(tensor, torch.Tensor)
            and tensor.device.type == "npu"
            and tensor.device.index is not None
        )
        assert len(device_idx_set) <= 1, f"Can not mix devices {device_idx_set}"
        device_interface = get_interface_for_device("npu")
        device_idx = (
            next(iter(device_idx_set))
            if device_idx_set
            else device_interface.current_device()
        )
        with device_interface.device(device_idx):
            result = benchmarker.benchmark_gpu(fn, device_type="npu")
            device_interface.synchronize()
        return result

    def make_run_fn(
        self,
        *input_tensors: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> Callable[[], None]:
        if out is None:
            out = output_tensor
        elif output_tensor is not None and output_tensor is not out:
            raise ValueError("out and output_tensor must refer to the same tensor")
        if out is None:
            raise ValueError("benchmark output tensor is required")

        mod = PyCodeCache.load_by_key_path(
            self.module_cache_key,
            self.module_path,
            set_sys_modules=False,
        )
        self._benchmark_module = mod
        run_method = getattr(mod, self.kernel_name).run
        run_method.__self__.with_bandwidth_info = False

        warmup_arg = {}
        if "warmup" in inspect.signature(run_method).parameters:
            warmup_arg["warmup"] = False

        if out.device.type == "cpu":
            stream = 0
        else:
            device_interface = get_interface_for_device(out.device.type)
            stream = device_interface.get_raw_stream(self.output_tensor_meta.device.index)

        launch_args = [*input_tensors]
        if self.kernel_has_output_arg:
            launch_args.append(out)
        launch_args.extend(self.extra_args)
        if isinstance(
            getattr(mod, self.kernel_name),
            torch._inductor.runtime.triton_heuristics.DebugAutotuner,
        ):
            return functools.partial(
                run_method,
                *launch_args,
                **warmup_arg,
                stream=stream,
            )
        return functools.partial(
            run_method,
            *launch_args,
            **warmup_arg,
            stream=stream,
            benchmark_run=True,
        )


class NPUTritonBenchmarkRequest(
    _NPUTritonBenchmarkRequestMixin,
    TritonGPUBenchmarkRequest,
):
    pass


class NPUFlexAttentionDkdvTemplateBuffer(ir.TritonTemplateBuffer):
    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        runtime_renderer_factory,
        dispatch_spec,
        mutated_inputs=None,
        allowed_prologue_inps=None,
    ):
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        self.runtime_renderer_factory = runtime_renderer_factory
        self.dispatch_spec = dispatch_spec


class NPUFlexAttentionDkdvTemplateCaller(TritonTemplateCaller):
    def __init__(
        self,
        *args,
        runtime_renderer_factory,
        dispatch_spec,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.runtime_renderer_factory = runtime_renderer_factory
        self.dispatch_spec = dispatch_spec

    def output_node(self):
        buffer = NPUFlexAttentionDkdvTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            runtime_renderer_factory=self.runtime_renderer_factory,
            dispatch_spec=self.dispatch_spec,
            mutated_inputs=self.mutated_inputs,
            allowed_prologue_inps=self.allowed_prologue_inps,
        )
        return ir.TensorBox.create(buffer)


@dataclasses.dataclass(frozen=True)
class NPUTritonRuntimeRenderer:
    kernel: Any
    render: Callable
    runtime_arg_names: dict[str, str]
    runtime_arg_dtypes: dict[str, torch.dtype]
    runtime_arg_buffers: dict[str, ir.Buffer]

    @contextlib.contextmanager
    def patch_runtime_args(self):
        graph_get_dtype = V.graph.get_dtype
        graph_get_buffer = V.graph.get_buffer
        scheduler = getattr(V.graph, "scheduler", None)
        scheduler_get_buffer_layout = (
            getattr(scheduler, "get_buffer_layout", None)
            if scheduler is not None
            else None
        )

        def get_dtype(name):
            if name in self.runtime_arg_dtypes:
                return self.runtime_arg_dtypes[name]
            return graph_get_dtype(name)

        def get_buffer(name):
            if name in self.runtime_arg_buffers:
                return self.runtime_arg_buffers[name]
            return graph_get_buffer(name)

        def get_buffer_layout(name):
            if name in self.runtime_arg_buffers:
                return self.runtime_arg_buffers[name].get_layout()
            return scheduler_get_buffer_layout(name)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(V.graph, "get_dtype", get_dtype))
            stack.enter_context(patch.object(V.graph, "get_buffer", get_buffer))
            if scheduler_get_buffer_layout is not None:
                stack.enter_context(
                    patch.object(
                        scheduler,
                        "get_buffer_layout",
                        get_buffer_layout,
                    )
                )
            yield

    def python_argdefs(self):
        with self.patch_runtime_args():
            return self.kernel.args.python_argdefs()


class NPUTritonTemplate(TritonTemplate):
    """NPU-specific Triton template for kernel generation.

    This class extends TritonTemplate to provide NPU-specific optimizations
    and configurations for Triton kernel generation.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        grid: Any,
        source: str,
        debug: bool = False,
        manual_output_buffer: Optional[str] = None,
        codegen_kernel_name: Optional[str] = None,
        compile_options: Optional[NPUTemplateCompileOption] = None,
    ) -> None:
        """Initialize NPU Triton template.

        Args:
            name: Template name for identification
            grid: Grid function for kernel launch configuration
            source: Triton kernel source code
            debug: Enable debug mode for verbose output
        """
        super().__init__(name, grid, source, debug)
        self.manual_output_buffer = manual_output_buffer
        self.codegen_kernel_name = codegen_kernel_name or f"triton_{name}"
        self.compile_options = compile_options or NPUTemplateCompileOption()

    def _write_index_dtype_define(
        self,
        defines: StringIO,
        numel: sympy.Expr,
        buffers: Any,
    ) -> None:
        can_use_32bit_indexing = TritonScheduling.can_use_32bit_indexing(
            numel, buffers
        )
        is_flex_attention = self.name.startswith("flex_attention")
        if not can_use_32bit_indexing and not is_flex_attention:
            raise NotImplementedError(
                "64-bit indexing is not yet implemented for triton templates"
            )
        if is_flex_attention:
            index_dtype = "tl.int32" if can_use_32bit_indexing else "tl.int64"
            defines.write(f"INDEX_DTYPE : tl.constexpr = {index_dtype}\n")

    def make_runtime_renderer_factory(
        self,
        *,
        input_nodes,
        runtime_args,
        layout,
        num_stages,
        num_warps,
        call_sizes=None,
        subgraphs=None,
        reset_to_zero_arg_names=None,
        **kwargs,
    ):
        runtime_args = tuple(runtime_args)
        call_sizes = list(call_sizes or layout.size)
        meta = dict(kwargs)
        self.compile_options.apply(meta)
        meta["ALLOW_TF32"] = "False"
        defines = StringIO()
        compile_option_keys = frozenset(self.compile_options.options)
        for name, value in meta.items():
            if name in compile_option_keys:
                continue
            if self.name.startswith("flex_attention") and name == "generate_with_caching":
                continue
            defines.write(f"{name} : tl.constexpr = {value}\n")
        fake_out = ir.Buffer(name="buf_out", layout=layout)
        numel = sympy_product(layout.size)
        if self.manual_output_buffer is None:
            buffers = itertools.chain(input_nodes, (fake_out,))
        else:
            buffers = input_nodes
            numel = sympy_product(call_sizes)
        self._write_index_dtype_define(defines, numel, buffers)
        kernel_options = {
            "defines": defines.getvalue(),
            "num_stages": num_stages,
            "num_warps": num_warps,
            "grid_fn": self.grid,
            "meta": meta,
            "call_sizes": call_sizes,
            "prefix_args": 0,
            "suffix_args": 0,
            "epilogue_fn": identity,
            "subgraphs": subgraphs,
            "always_freeze_layout": getattr(
                self, "always_freeze_layout", False
            ),
            "manual_output_buffer": self.manual_output_buffer,
            "reset_to_zero_arg_names": reset_to_zero_arg_names,
            "compile_option_keys": compile_option_keys,
        }

        def create_renderer(out_node):
            runtime_nodes = [
                ir.Buffer(
                    name=f"__npu_runtime_{runtime_arg.name.lower()}",
                    layout=ir.FixedLayout(
                        out_node.get_device(),
                        runtime_arg.dtype,
                        [1] * runtime_arg.rank,
                    ),
                )
                for runtime_arg in runtime_args
            ]
            all_input_nodes = [*input_nodes, *runtime_nodes]
            runtime_arg_names = {
                node.get_name(): runtime_arg.wrapper_name
                for node, runtime_arg in zip(runtime_nodes, runtime_args)
            }
            runtime_arg_dtypes = {
                node.get_name(): runtime_arg.dtype
                for node, runtime_arg in zip(runtime_nodes, runtime_args)
            }
            kernel = NPUTritonTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
                input_nodes=all_input_nodes,
                output_node=out_node,
                workspace_arg=None,
                use_jit=False,
                **kernel_options,
            )

            def render():
                with patch.object(
                    V.graph,
                    "get_dtype",
                    self._fake_get_dtype([out_node, *runtime_nodes]),
                ):
                    return kernel.render(self.template, meta)

            return NPUTritonRuntimeRenderer(
                kernel=kernel,
                render=render,
                runtime_arg_names=runtime_arg_names,
                runtime_arg_dtypes=runtime_arg_dtypes,
                runtime_arg_buffers={
                    node.get_name(): node for node in runtime_nodes
                },
            )

        return create_renderer

    def generate(
        self,
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        num_stages: int,
        num_warps: int,
        num_consumer_groups: int = 0,
        num_buffers_warp_spec: int = 0,
        prefix_args: int = 0,
        suffix_args: int = 0,
        epilogue_fn: Callable = identity,
        epilogue_fn_hash: Optional[str] = None,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        mutated_inputs: Optional[list[ir.IRNode]] = None,
        call_sizes: Optional[list[sympy.Expr]] = None,
        workspace_arg: Optional[Any] = None,
        generate_with_caching: bool = False,
        hint_override: Optional[int] = None,
        tma_store: bool = False,
        tma_load_for_template_epilogue: bool = False,
        transpose_discontiguous_tensor_descriptors_override: Optional[bool] = None,
        triton_meta: Optional[dict[str, Any]] = None,
        reset_to_zero_arg_names: Optional[list[str]] = None,
        large_input_buffers: Optional[list[ir.IRNode]] = None,
        runtime_renderer_factory: Optional[Callable] = None,
        dispatch_spec: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[ir.ChoiceCaller]:
        kwargs = dict(kwargs)
        self.compile_options.apply(kwargs)
        compile_option_keys = frozenset(self.compile_options.options)
        defines = StringIO()
        kwargs["ALLOW_TF32"] = "False"
        for name, val in kwargs.items():
            if name in compile_option_keys:
                continue
            if self.name.startswith("flex_attention") and name == "generate_with_caching":
                continue
            defines.write(f"{name} : tl.constexpr = {val}\n")

        fake_out = ir.Buffer(name="buf_out", layout=layout)
        kernel_name = f"triton_{self.name}"

        numel = sympy_product(layout.size)
        large_input_buffer_names = {
            buffer.get_name() for buffer in large_input_buffers or []
        }
        checked_input_nodes = [
            input_node
            for input_node in input_nodes
            if input_node.get_name() not in large_input_buffer_names
        ]
        if self.manual_output_buffer is None:
            buffers = itertools.chain(checked_input_nodes, (fake_out,))
        else:
            buffers = checked_input_nodes
            numel = sympy_product(call_sizes or layout.size)
        self._write_index_dtype_define(defines, numel, buffers)

        if not self.name.startswith("flex_attention"):
            defines.write("INDEX_DTYPE : tl.constexpr = tl.int32\n")
        defines = defines.getvalue()

        if call_sizes is None:
            call_sizes = layout.size

        kernel_options = {
            "input_nodes": input_nodes,
            "defines": defines,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "num_consumer_groups": num_consumer_groups,
            "num_buffers_warp_spec": num_buffers_warp_spec,
            "grid_fn": self.grid,
            "meta": kwargs,
            "call_sizes": call_sizes,
            "prefix_args": prefix_args,
            "suffix_args": suffix_args,
            "epilogue_fn": epilogue_fn,
            "subgraphs": subgraphs,
            "always_freeze_layout": getattr(
                self, "always_freeze_layout", False
            ),
            "tma_store": tma_store,
            "tma_load_for_template_epilogue": tma_load_for_template_epilogue,
            "transpose_discontiguous_tensor_descriptors_override": (
                transpose_discontiguous_tensor_descriptors_override
            ),
            "hint_override": hint_override,
            "triton_meta": triton_meta,
            "manual_output_buffer": self.manual_output_buffer,
            "reset_to_zero_arg_names": reset_to_zero_arg_names,
            "compile_option_keys": compile_option_keys,
        }

        with (
            patch.object(V.graph, "get_dtype", self._fake_get_dtype(fake_out)),
            V.graph.set_current_device(layout.device),
            NPUTritonTemplateKernel(
                kernel_name=kernel_name,
                output_node=fake_out,
                workspace_arg=workspace_arg,
                use_jit=False,
                **kernel_options,
            ) as kernel,
        ):
            try:
                template = kernel.render(self.template, kwargs)
                if "<STORE_OUTPUT>" not in kernel.subgraph_bodies:
                    with kernel.create_subgraph_body("<STORE_OUTPUT>"):
                        pass

                    def empty_hook():
                        return ""

                    kernel.render_hooks["<STORE_OUTPUT>"] = empty_hook
                with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                    code = template.finalize_all()
            except ZeroDivisionError:
                log.debug(
                    "ZeroDivisionError during kernel rendering for %s, "
                    "returning None to skip this configuration",
                    kernel_name,
                )
                return None

            if self.debug:
                log.debug("Generated Code:\n", code)
            # Build extra string for cache key and description.
            # We use '-' as the intermediate separator instead of ', ' because some
            # kwargs values (like call_sizes, mutated_inputs, subgraphs) contain
            # commas in their repr() output. Using '-' avoids ambiguity when parsing.
            # The trailing '-' ensures consistent formatting before strip/replace.
            extra = (
                "-".join(
                    [
                        *[
                            f"{kwarg}={repr(kwargs[kwarg])}"
                            for kwarg in sorted(kwargs.keys())
                        ],
                        f"num_stages={num_stages}",
                        f"num_warps={num_warps}",
                    ]
                )
                + "-"
            )
            mod = PyCodeCache.load(code, extra)

        input_call_args = tuple(kernel.args.input_buffers.keys())

        # We expect the input_buffer order to be [*input_nodes, *captured_buffers]
        expected_input_args = tuple(unique(x.get_name() for x in input_nodes))
        assert input_call_args[: len(expected_input_args)] == expected_input_args, (
            input_call_args,
            expected_input_args,
        )

        full_input_nodes = tuple([V.graph.get_buffer(k) for k in input_call_args])
        extra_args = V.graph.sizevars.optimization_hints(
            map(sympy.expand, tuple(kernel.args.sizevars.keys())),
            fallback=config.unbacked_symint_fallback,
        )
        kernel_has_output_arg = any(
            outer not in kernel.args.inplace_buffers
            and not isinstance(inner, RemovedArg)
            for outer, inner in kernel.args.output_buffers.items()
        )

        kernel_hash_name = f"triton_{self.name}_{next(self.index_counter)}"

        def make_kernel_render(out_node):
            kernel = NPUTritonTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
                output_node=out_node,
                workspace_arg=workspace_arg,
                use_jit=False,
                **kernel_options,
            )

            render = functools.partial(
                kernel.render,
                self.template,
                kwargs,
            )
            return kernel, render

        # create the BenchmarkRequest
        assert mod.__file__ is not None
        grid = self.grid(
            *V.graph.sizevars.optimization_hints(
                call_sizes,
                fallback=config.unbacked_symint_fallback,
            ),
            kwargs,
        )
        bmreq_cls: type[TritonBenchmarkRequest]
        if layout.device.type == "cpu":
            bmreq_cls = TritonCPUBenchmarkRequest
        else:
            bmreq_cls = NPUTritonBenchmarkRequest
        bmreq = bmreq_cls(
            module_path=mod.__file__,
            module_cache_key=mod.key,
            kernel_name=kernel_name,
            extra_args=[*extra_args, *grid],
            num_stages=num_stages,
            num_warps=num_warps,
            matrix_instr_nonkdim=kwargs.get("matrix_instr_nonkdim", 0),
            waves_per_eu=kwargs.get("waves_per_eu", 0),
            kpack=kwargs.get("kpack", 2),
            input_tensor_meta=TensorMeta.from_irnodes(full_input_nodes),  # type: ignore[arg-type]
            output_tensor_meta=TensorMeta.from_irnodes(layout),
        )
        bmreq.kernel_has_output_arg = kernel_has_output_arg

        caller_type = TritonTemplateCaller
        caller_kwargs = {}
        if runtime_renderer_factory is not None:
            assert dispatch_spec is not None
            caller_type = NPUFlexAttentionDkdvTemplateCaller
            caller_kwargs = {
                "runtime_renderer_factory": runtime_renderer_factory,
                "dispatch_spec": dispatch_spec,
            }

        return caller_type(
            kernel_hash_name,
            full_input_nodes,
            layout,
            make_kernel_render,
            # Convert '-' back to ', ' for human-readable description in logs.
            # Note: This assumes kwarg values don't contain '-' characters.
            # See the comment above for why '-' was used as separator.
            extra.strip("-").replace("-", ", "),
            bmreq,
            log_info={
                "tile_shape": str(
                    (
                        kwargs.get("BLOCK_M", -1),
                        kwargs.get("BLOCK_K", -1),
                        kwargs.get("BLOCK_N", -1),
                    )
                ),
                "num_stages": num_stages,
                "num_warps": num_warps,
                "allow_tf32": str(kwargs.get("ALLOW_TF32", None)),
                "acc_type": str(kwargs.get("ACC_TYPE", None)),
            },
            mutated_inputs=mutated_inputs,
            workspace_arg=workspace_arg,
            allowed_prologue_inps=kernel.prologue_supported_inputs.copy(),
            hint_override=hint_override,
            **caller_kwargs,
        )


class NPUTritonTemplateKernel(TritonTemplateKernel):
    """NPU-specific Triton template kernel for code generation.

    This class extends TritonTemplateKernel to provide NPU-specific
    kernel generation and compilation functionality.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[ir.IRNode],
        output_node: ir.IRNode,
        defines: str,
        num_stages: int,
        num_warps: int,
        grid_fn: Callable,
        meta: dict[str, Any],
        call_sizes: list[sympy.Expr],
        num_consumer_groups: int = 0,
        num_buffers_warp_spec: int = 0,
        use_jit: bool = False,
        tma_store: bool = False,
        tma_load_for_template_epilogue: bool = False,
        transpose_discontiguous_tensor_descriptors_override: Optional[bool] = None,
        prefix_args: int = 0,
        suffix_args: int = 0,
        epilogue_fn: Callable = identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[Any] = None,
        prologue_loads_all_inputs: bool = False,
        hint_override: Optional[int] = None,
        triton_meta: Optional[dict[str, Any]] = None,
        always_freeze_layout: bool = False,
        index_dtype_override: Optional[str] = None,
        manual_output_buffer: Optional[str] = None,
        reset_to_zero_arg_names: Optional[list[str]] = None,
        compile_option_keys: frozenset[str] = frozenset(),
    ) -> None:
        """Initialize NPU Triton template kernel.

        Args:
            kernel_name: Name of the kernel
            input_nodes: List of input IR nodes
            output_node: Output IR node
            defines: Kernel defines string
            num_stages: Number of pipeline stages
            num_warps: Number of warps
            grid_fn: Grid function for launch configuration
            meta: Metadata dictionary
            call_sizes: Call sizes for grid computation
            use_jit: Whether to use JIT compilation
            prefix_args: Number of prefix arguments
            suffix_args: Number of suffix arguments
            epilogue_fn: Epilogue function
            subgraphs: List of subgraph buffers
            workspace_arg: Workspace argument
        """
        super().__init__(
            kernel_name=kernel_name,
            input_nodes=input_nodes,
            output_node=output_node,
            defines=defines,
            num_stages=num_stages,
            num_warps=num_warps,
            grid_fn=grid_fn,
            meta=meta,
            call_sizes=call_sizes,
            num_consumer_groups=num_consumer_groups,
            num_buffers_warp_spec=num_buffers_warp_spec,
            use_jit=use_jit,
            tma_store=tma_store,
            tma_load_for_template_epilogue=tma_load_for_template_epilogue,
            transpose_discontiguous_tensor_descriptors_override=(
                transpose_discontiguous_tensor_descriptors_override
            ),
            prefix_args=prefix_args,
            suffix_args=suffix_args,
            epilogue_fn=epilogue_fn,
            subgraphs=subgraphs,
            workspace_arg=workspace_arg,
            prologue_loads_all_inputs=prologue_loads_all_inputs,
            hint_override=hint_override,
            triton_meta=triton_meta,
            always_freeze_layout=always_freeze_layout,
            index_dtype_override=index_dtype_override,
        )
        self.manual_output_buffer = manual_output_buffer
        self.reset_to_zero_arg_names = reset_to_zero_arg_names
        self.compile_option_keys = compile_option_keys

    def create_cse_var(self, name=None, bounds=None, dtype=None, shape=None, **kwargs):
        # torch>=2.8 added an assertion in TritonCSEVariable.__init__ requiring
        # shape to be non-None. Some upstream codegen paths (e.g.
        # online_softmax_reduce) call newvar(dtype=...) without shape.
        # Provide a scalar default so the assertion is satisfied; the actual
        # shape is back-filled later by CSE.generate (see common.py).
        if shape is None:
            shape = ()
        return super().create_cse_var(name, bounds, dtype, shape, **kwargs)

    def _register_output_buffer(self, arg_name: str) -> None:
        self.args.output_buffers.setdefault(self.output_node.get_name(), arg_name)

    def def_kernel(self, *argnames: str) -> str:
        """Hook called from template code to generate function def and needed args.

        Args:
            *argnames: Variable number of argument names

        Returns:
            Render hook key string
        """
        assert all(isinstance(x, str) for x in argnames)
        renames = IndentedBuffer(initial_indent=1)

        named_args = self.input_nodes[
            self.prefix_args : len(self.input_nodes) - self.suffix_args
        ]

        assert len(argnames) == len(named_args), (
            len(argnames),
            len(named_args),
            self.prefix_args,
            len(self.input_nodes),
        )

        # Unified processing of all input nodes
        for idx, input_node in enumerate(self.input_nodes):
            node_name = input_node.get_name()

            # Skip removed or fused buffers
            if node_name in V.graph.removed_buffers:
                continue
            if node_name in self.prologue_fused_inputs:
                continue

            # Process prefix args
            if idx < self.prefix_args:
                self.args.input(node_name)
            # Process named args
            elif idx < len(self.input_nodes) - self.suffix_args:
                name = argnames[idx - self.prefix_args]
                arg_name = f"arg_{name}"
                self.named_input_nodes[name] = input_node
                self.args.input_buffers[node_name] = arg_name
            # Process suffix args
            else:
                self.args.input(node_name)

        # The args may be duplicated, so renaming must be after args are de-duplicated.
        for name in argnames:
            input_node = self.named_input_nodes[name]
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue
            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f"{name} = {arg_name}")
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        def hook():
            # python_argdefs() cannot be run until after the rest of the template lazily adds more args
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.splice(_gen_npu_template_triton_imports())
            code.splice(self.jit_lines())
            code.writeline(
                f"def {self.kernel_name}({', '.join(x.full_name() for x in arg_defs)}):"
            )
            with code.indent():
                code.splice(self.defines)
                code.splice(renames.getvalue())
            return code.getvalue()

        assert "<DEF_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_KERNEL>"] = hook
        if self.manual_output_buffer is not None:
            self._register_output_buffer(self.manual_output_buffer)
        if len(argnames) == 0 and len(self.input_nodes) == 0:
            self._register_output_buffer("out_ptr0")
        return "<DEF_KERNEL>"


    def _get_store_output_subgraph_name(self, i: int) -> str:
        """Override to use a fixed name without index suffix.

        The NPU codegen (scheduling.py codegen_template and generate method)
        expects the store_output subgraph body to be named ``<STORE_OUTPUT>``
        (without the ``_{i}`` suffix used by the upstream SIMDKernel).
        """
        return "<STORE_OUTPUT>"


    def jit_lines(self) -> str:
        from torch._inductor.codegen.triton import TritonKernel
        from torch._inductor.codegen.triton_utils import (
            config_of,
            equal_1_arg_indices,
            signature_to_meta,
        )
        from torch._inductor.runtime.hints import DeviceProperties
        from torch._inductor.runtime.triton_heuristics import FixedGrid

        if self.use_jit:
            return "@triton.jit"

        argdefs, _, signature, _ = self.args.python_argdefs()
        triton_meta = {
            "signature": signature_to_meta(
                signature,
                size_dtype=self.index_dtype,
                argdefs=argdefs,
            ),
            "device": DeviceProperties.create(self.output_node.get_device()),
            "constants": {},
        }
        triton_meta["configs"] = [config_of(signature)]
        if self.reset_to_zero_arg_names:
            triton_meta["reset_to_zero"] = self.reset_to_zero_arg_names
        for arg_num in equal_1_arg_indices(signature):
            triton_meta["constants"][signature[arg_num].name] = 1

        matrix_instr_nonkdim = self.meta.get("matrix_instr_nonkdim", None)
        waves_per_eu = self.meta.get("waves_per_eu", None)
        kpack = self.meta.get("kpack", None)
        if matrix_instr_nonkdim:
            triton_meta["matrix_instr_nonkdim"] = matrix_instr_nonkdim
        if waves_per_eu:
            triton_meta["waves_per_eu"] = waves_per_eu
        if kpack:
            triton_meta["kpack"] = kpack

        _add_npu_template_compile_options_to_triton_meta(
            triton_meta,
            self.meta,
            self.compile_option_keys,
        )
        self.triton_meta = triton_meta

        inductor_meta = {
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            **TritonKernel.inductor_meta_common(),
            **FixedGrid.setup_grid_as_args(),
        }

        if config.profile_bandwidth or config.benchmark_kernel:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        _add_npu_template_meta_to_inductor_meta(inductor_meta, self.meta)

        if npu_config.aggresive_autotune:
            inductor_meta["profile_bandwidth_with_do_bench_using_profiling"] = True

        return f"""
            @triton_heuristics.template(
                num_stages={self.num_stages},
                num_warps={self.num_warps},
                triton_meta={triton_meta!r},
                inductor_meta={inductor_meta!r},
            )
            @triton.jit
        """


def patch_algorithm_selector() -> None:
    """Patch AlgorithmSelectorCache with NPU-specific implementations.

    This function replaces the default AlgorithmSelectorCache methods with
    NPU-optimized versions that include profiling and benchmarking capabilities
    specific to NPU hardware.
    """

    def __call__(
        self,
        name: str,
        choices: List[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[Dict[int, Callable[[ir.Buffer], torch.Tensor]]] = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template: bool = False,
        defer_epilogue_compile_only: bool = False,
    ) -> Any:
        from .codegen.catlass.catlass_kernel import CATLASSTemplateCaller

        defer_epilogue_to_scheduler = (
            defer_epilogue_compile_only
            and return_multi_template
            and input_gen_fns is not None
        )

        # Templates selected with input_gen_fns require specific input data to avoid IMA.
        # FlexAttention keeps using those generators for its lowering-time autotune,
        # then defers only the final, epilogue-aware compilation choice.
        # TODO(jgong5): support multi-template on CPU
        if layout.device.type == "cpu" or (
            input_gen_fns is not None and not defer_epilogue_to_scheduler
        ):
            return_multi_template = False

        choices = [choice for choice in choices if choice is not None]
        successful_precompile_choice_hashes: set[str] = set()
        select_first_compilable_only = bool(choices) and all(
            getattr(choice, "_nobench_select_first_compilable", False)
            for choice in choices
        )

        if mm_file_name := get_mm_log_filename():
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]
            append_to_log(mm_file_name, {"invoke": str((M, K, N))})

        if len(choices) == 0:
            backend_config = (
                "max_autotune_gemm_backends"
                if name != "convolution"
                else "max_autotune_conv_backends"
            )
            raise NoValidChoicesError(
                f"No choices to select, please consider adding ATEN into {backend_config} "
                "config (defined in torch/_inductor/config.py) to allow at least one choice. "
            )
        log.debug("Max autotune selects from %s choices.", str(len(choices)))

        if (
            len(choices) == 1
            and not select_first_compilable_only
            and not defer_epilogue_compile_only
        ):
            if not isinstance(choices[0], CATLASSTemplateCaller):
                # CATLASSTemplateCaller still needs to go through autotuning process to retrieve workspace size.
                return choices[0].output_node(), choices[0]

        @functools.lru_cache(None)
        def make_benchmark_fn():
            return self.make_benchmark_fn(choices, input_nodes, layout, input_gen_fns)

        inputs_key = create_inputs_key(input_nodes)

        def precompile(choices) -> Callable[[], None]:
            log.debug("Starting precompilation")

            def no_op(*args, **kwargs):
                return

            if (
                precompilation_timeout_seconds is None
                or precompilation_timeout_seconds <= 0
            ):
                return no_op

            num_workers = min(get_num_workers(), len(choices))

            if num_workers <= 0:
                return no_op

            # NOTE: The upstream Python 3.11.0-3.11.8 guard is relaxed here
            # because NPU codegen already wraps each precompile task in
            # restore_stdout_stderr(), which prevents the stdout/stderr race
            # that the version check was protecting against.
            # Cap workers to npu_config.precompile_thread_num for
            # consistency with the triton_heuristics compile_thread_pool.
            try:
                from .. import config as npu_config
                num_workers = min(num_workers, npu_config.precompile_thread_num)
            except Exception:
                pass

            if not select_first_compilable_only:
                # check local and global cache before precompiling
                timings = self.lookup(
                    choices,
                    name,
                    inputs_key,
                    benchmark=None,
                )

                if timings:
                    # compilation in precompile stage is much cheaper than that in
                    # autotuning stage
                    if len(timings) == len(choices):
                        log.debug("Timings found in cache, returning no_op")
                        return no_op

                if config.search_autotune_cache and not (
                    config.max_autotune or config.max_autotune_gemm
                ):
                    return no_op

            precompile_key = create_precompile_key(name, inputs_key, choices)
            if not select_first_compilable_only and (
                precompile_func := self.precompile_cache.get(precompile_key)
            ):
                return precompile_func

            log.info(
                "Multithreaded precompilation for %d choices using %d worker threads",
                len(choices),
                num_workers,
            )

            # In rare circumstances, because python threads inherit global state,
            # thread pool executor can race and leave stdout/stderr in a state
            # different than the original values. we explicitly restore the state
            # here to avoid this issue.

            def precompile_with_captured_stdout(choice):
                log.debug("Precompiling choice with captured stdout: %s", choice)
                with restore_stdout_stderr():
                    choice.precompile()

            def on_complete(future):
                assert future in start_times
                elapsed_times[future] = time.time() - start_times[future]
                log.debug(
                    "Precompilation complete for future: %s, elapsed time: %.02fs",
                    future,
                    elapsed_times[future],
                )

            executor = ThreadPoolExecutor(max_workers=num_workers)
            async_compile = torch._inductor.async_compile.AsyncCompile()

            futures: dict[Future[Any], ChoiceCaller] = {}
            start_times: dict[Future[Any], float] = {}
            elapsed_times: dict[Future[Any], float] = {}

            # Some choices only differ in runtime arguments, so we
            # skip a choice if it has the same hash as a previously seen choice
            seen_choices: OrderedSet[ChoiceCaller] = OrderedSet()
            for c in choices:
                # Skip choices which we have already issued a precompile
                if c.hash_key() in seen_choices:
                    log.debug("Skipping already seen choice: %s", c)
                    continue
                else:
                    seen_choices.add(c.hash_key())

                if hasattr(c, "precompile"):
                    future = executor.submit(precompile_with_captured_stdout, c)
                    log.debug("Submitted precompile for choice: %s", c)

                    start_times[future] = time.time()
                    future.add_done_callback(on_complete)
                    futures[future] = c

            @functools.lru_cache(None)
            @restore_stdout_stderr()
            def wait_on_futures():
                counters["inductor"]["select_algorithm_precompile"] += 1
                for future in as_completed(
                    futures,
                    timeout=precompilation_timeout_seconds,
                ):
                    if e := future.exception():
                        log.error(
                            "Exception %s for benchmark choice %s", e, futures[future]
                        )
                    else:
                        successful_precompile_choice_hashes.add(
                            futures[future].hash_key()
                        )
                        counters["inductor"]["select_algorithm_num_precompiles"] += 1
                        log.info(
                            "Precompiling benchmark choice %s took %.02fs",
                            _format_choice_debug_label(futures[future]),
                            elapsed_times[future],
                        )

                executor.shutdown(wait=True)

            if not select_first_compilable_only:
                self.precompile_cache[precompile_key] = wait_on_futures

            return wait_on_futures

        def autotune(choices):
            log.debug("Starting autotuning")
            with dynamo_timed(
                f"{name}_template_autotuning",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="compile_time_autotune_time_us",
            ):
                return make_benchmark_fn()(choices)

        if config.autotune_in_subproc:
            from torch._inductor.autotune_process import tuning_pool

            # do the optional warmup
            tuning_pool.initialize()

        def do_autotuning(precompile_fn):
            precompile_start_ts = time.time()
            with dynamo_timed(
                f"{name}_template_precompiling",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="compile_time_autotune_time_us",
            ):
                precompile_fn()
            precompile_elapse = time.time() - precompile_start_ts

            autotune_start_ts = time.time()
            timings = self.lookup(
                choices,
                name,
                inputs_key,
                autotune,
            )
            autotune_elapse = time.time() - autotune_start_ts
            log.debug("Autotuning elapsed time: %.02fs", autotune_elapse)

            if timings and all(
                not math.isfinite(timing) for timing in timings.values()
            ):
                raise NoValidChoicesError

            if make_benchmark_fn.cache_info().currsize:
                counters["inductor"]["select_algorithm_autotune"] += 1

            if (
                make_benchmark_fn.cache_info().currsize
                or log.getEffectiveLevel() == logging.DEBUG
                or config.trace.log_autotuning_results
            ):
                self.log_results(
                    name, input_nodes, timings, autotune_elapse, precompile_elapse
                )

            for feedback_fn in self.feedback_saver_fns:
                feedback_fn(timings, name, input_nodes, choices)

            return timings

        if select_first_compilable_only:
            counters["inductor"]["select_algorithm_precompile"] += 1
            selected_choice = None
            for choice in choices:
                try:
                    with restore_stdout_stderr():
                        choice.precompile()
                    successful_precompile_choice_hashes.add(choice.hash_key())
                    counters["inductor"]["select_algorithm_num_precompiles"] += 1
                    selected_choice = choice
                    break
                except Exception as e:
                    log.warning(
                        "Compile Fail for no-benchmark choice %s "
                        "during ordered precompile",
                        _format_choice_debug_label(choice),
                    )
                    log.debug(  # noqa: G200
                        "Exception %s for no-benchmark choice %s "
                        "during ordered precompile",
                        e,
                        _format_choice_debug_label(choice),
                    )
            if selected_choice is None:
                raise NoValidChoicesError(
                    f"No compilable choices found for {name} in no-benchmark mode."
                )
            log.info(
                "[select_algorithm] No-benchmark selected first compilable choice %s",
                _format_choice_debug_label(selected_choice),
            )
            return selected_choice.output_node(), selected_choice

        precompile_fn = precompile(choices)

        if return_multi_template and (config.max_autotune or config.max_autotune_gemm):

            def get_timings(hint_override=None):
                timings = do_autotuning(precompile_fn)
                min_extern_choice = float("inf")
                for choice, timing in timings.items():
                    if isinstance(choice, ExternKernelCaller):
                        min_extern_choice = min(min_extern_choice, timing)

                timings = {
                    choice: time
                    for choice, time in timings.items()
                    if (
                        time <= min_extern_choice
                        or not isinstance(choice, ExternKernelCaller)
                    )
                }

                return timings

            # We take the union of allowed prologue inputs from all choices,
            # and, within benchmark fusion, don't allow prologue fusion for
            # choices which dont support the whole union.
            allowed_prologue_inps: OrderedSet[str] = OrderedSet()
            for c in choices:
                if isinstance(c, TritonTemplateCaller):
                    allowed_prologue_inps |= c.allowed_prologue_inps

            multi_template = torch._inductor.ir.MultiTemplateBuffer(
                layout,
                input_nodes,
                get_timings,
                choices,
                allowed_prologue_inps,
            )
            if defer_epilogue_to_scheduler:
                multi_template._npu_deferred_epilogue_compile_only = True
            return torch._inductor.ir.TensorBox.create(multi_template), None

        timings = do_autotuning(precompile_fn)
        if timings == {} or choices[0] not in timings:
            fallback_choice = _select_first_usable_choice_in_order(
                choices,
                timings,
                successful_precompile_choice_hashes,
            )
            if fallback_choice is not None:
                log.info(
                    "[select_algorithm] Fallback selected choice %s",
                    _format_choice_debug_label(fallback_choice),
                )
                return fallback_choice.output_node(), fallback_choice
            log.warning(
                "[select_algorithm] No timings and no usable fallback; "
                "returning first choice %s",
                _format_choice_debug_label(choices[0]),
            )
            return choices[0].output_node(), choices[0]

        selected_key = builtins.min(timings, key=timings.__getitem__)
        selected_choice = selected_key.output_node()
        log.debug("selected choice: %s", str(selected_choice))
        return selected_choice, selected_key

    @classmethod
    def make_benchmark_fn(
        cls,
        choices: List[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[Dict[int, Callable[[ir.Buffer], torch.Tensor]]] = None,
    ) -> Callable:
        """Create a benchmark function for the given choices.

        Args:
            choices: List of choice callers to benchmark
            input_nodes: List of input IR nodes
            layout: Output layout
            input_gen_fns: Optional dict mapping arg indices to input generation functions

        Returns:
            Benchmark function that can be called with choices
        """
        if input_gen_fns is None:
            input_gen_fns = {}

        def get_inputs(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]],
        ) -> AutotuneArgs:
            # de-duplicate args
            unique_example_inputs = {
                x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
                for i, x in enumerate(input_nodes)
            }
            example_inputs = list(unique_example_inputs.values())
            example_inputs_extern = [
                (
                    unique_example_inputs[input_node.get_name()]
                    if unique_example_inputs[input_node.get_name()].is_mkldnn
                    else torch.as_strided(
                        unique_example_inputs[input_node.get_name()],
                        V.graph.sizevars.optimization_hints(
                            input_node.get_size(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                        V.graph.sizevars.optimization_hints(
                            input_node.get_stride(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                        V.graph.sizevars.optimization_hint(
                            input_node.get_layout().offset,
                            fallback=config.unbacked_symint_fallback,
                        ),
                    )
                )
                for input_node in input_nodes
            ]

            from .codegen.catlass.catlass_kernel import CATLASSTemplateCaller
            is_group_mm = False
            for choice in choices:
                if isinstance(choice, CATLASSTemplateCaller) and "GroupedMatmulSliceMTla" in choice.description:
                    is_group_mm = True

            if not is_group_mm and len(input_nodes) == 3:
                # reorder inputs here because addmm catlass template
                # expects (x, w, bias) but torch is bias, x, w
                example_inputs = example_inputs[1:] + [example_inputs[0]]
            out = cls.benchmark_example_value(layout)
            out_extern = torch.as_strided(
                out,
                out.size(),
                out.stride(),
                V.graph.sizevars.optimization_hint(
                    layout.offset,
                    fallback=config.unbacked_symint_fallback,
                ),
            )
            expected = None
            if VERIFY:
                choices[0].benchmark(*example_inputs_extern, out=out_extern)
                expected = out_extern.clone()

            return AutotuneArgs.from_choice_args(
                example_inputs,
                example_inputs_extern,
                out,
                out_extern,
                expected,
            )

        if DEBUG:
            log.debug("%d tuning requests:", len(choices))

        def benchmark_choice_in_current_process(
            choice: ChoiceCaller, autotune_args: AutotuneArgs
        ) -> float:
            is_extern = isinstance(choice, ExternKernelCaller)
            benchmark_tensors = autotune_args.get_benchmark_tensors(is_extern)
            inpts, output = benchmark_tensors.unpack()
            output.zero_()
            result = choice.benchmark(*inpts, out=output)
            if VERIFY and autotune_args.expected is not None:
                autotune_args.verify(**VERIFY)
            if torch.npu.is_available():
                torch.npu.synchronize()  # shake out any NPU errors
            return result

        def profiling_choices_in_current_process(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]],
        ) -> Dict[Union[ExternKernelCaller, TritonTemplateCaller], float]:
            inputs = get_inputs(choices)
            funcs = []
            for choice in choices:
                is_extern = isinstance(choice, ExternKernelCaller)
                benchmark_tensors = inputs.get_benchmark_tensors(is_extern)
                inpts, output = benchmark_tensors.unpack()
                output.zero_()
                if is_extern:
                    algo = choice.to_callable()
                    fn = algo
                    args = tuple(inpts)
                    kwargs = {"out": output}
                else:
                    # catlass & triton
                    fn = choice.bmreq.make_run_fn(*inpts, output_tensor=output)
                    args = ()
                    kwargs = {}
                funcs.append((fn, args, kwargs))

            # batch profiling all funcs in single profiler
            func_times = do_batch_profiling(funcs)
            return {choice: func_times[i] for i, choice in enumerate(choices)}

        def do_batch_profiling(
            funcs: List[Tuple[Callable, Tuple, Dict]], key: Optional[str] = None
        ) -> List[Optional[float]]:
            import torch_npu
            import shutil
            import uuid
            import hashlib

            def delete_file(base_path):
                if os.path.exists(base_path):
                    shutil.rmtree(base_path)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
                data_simplification=False,
            )

            random_uuid = uuid.uuid4().hex
            md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()

            num_funcs = len(funcs)
            torch_path = os.path.join(os.getcwd(), "profile_results", md5_hash)
            TOTAL_STEP = 50
            l2_cache_size = 192 * (1 << 20)
            buffer = torch.empty(l2_cache_size // 4, dtype=torch.int, device="npu")
            buffer = buffer.float()
            buffer.sum()
            torch.npu.synchronize()  # shake out of any npu error
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=tensorboard_trace_handler(torch_path),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            ):
                for fn, args, kwargs in funcs:
                    for _ in range(TOTAL_STEP):
                        buffer.sum()
                        fn(*args, **kwargs)
                        torch.npu.synchronize()
                    # One aclnn op may be separated into multiple ops, recorded in kernel_details.csv,
                    # which makes us hard to analyze the kernel_detail.csv. Therefore, an abs operation is added here,
                    # aiming to help us recognize different ops.
                    buffer.abs_()
                    torch.npu.synchronize()
            del buffer

            import pandas as pd

            for root, _, files in os.walk(torch_path):
                for file in files:
                    if file != "kernel_details.csv":
                        continue
                    target_file = os.path.join(root, file)
                    df = pd.read_csv(target_file)
                    # filter out l2 cache clear operation
                    filter_cond = ~df["Name"].str.contains(r"ReduceSum", case=False, na=False)
                    filter_df = df[filter_cond]
                    if key is not None:
                        key_rows = filter_df[filter_df["Name"].str.contains(key, na=False)]
                    else:
                        key_rows = filter_df
                    time_cost = []
                    last_df_index = -1
                    for idx, row in key_rows.iterrows():
                        if "absaicore" in row["Name"].lower():
                            time_cost.append(key_rows.loc[last_df_index + 1:idx - 1, 'Duration(us)'].sum())
                            last_df_index = idx
                    time_cost = [x / TOTAL_STEP / 1e3 for x in time_cost]
                    delete_file(torch_path)
                    return time_cost

            delete_file(torch_path)
            return []

        def benchmark_in_current_process(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]],
        ) -> Dict[Union[ExternKernelCaller, TritonTemplateCaller], float]:
            inputs = get_inputs(choices)
            timings = {}
            for choice in choices:
                try:
                    timing = benchmark_choice_in_current_process(choice, inputs)
                except NPUCompileError as e:
                    log.error(  # noqa: G200
                        "NPU compilation error during autotuning: \n%s. \nIgnoring this choice.",
                        str(e),
                    )
                    timing = float("inf")
                except NotImplementedError as e:
                    log.warning("Not yet implemented: %s", e)  # noqa: G200
                    timing = float("inf")
                except RuntimeError as e:
                    msg = str(e)
                    if "invalid argument" in msg:
                        msg += "\n\nThis may mean this NPU is too small for max_autotune mode.\n\n"
                    else:
                        if "illegal memory access" in msg:
                            msg += "\n\nEither error in template or triton bug.\n"
                    log.error(
                        "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
                        msg,
                    )
                    timing = float("inf")
                except AssertionError as e:
                    raise AssertionError(  # noqa: B904
                        f"Incorrect result from choice {choice}\n"
                    ) from e
                except Exception as e:
                    try:
                        from triton.runtime.autotuner import OutOfResources

                        if isinstance(e, OutOfResources):
                            log.warning("%s", e)  # noqa: G200
                            timing = float("inf")
                        else:
                            raise e
                    except ImportError:
                        raise e from None

                timings[choice] = timing

            return timings

        def benchmark_in_sub_process(
            choices: Union[List[ExternKernelCaller], List[TritonTemplateCaller]],
        ):
            from torch._inductor import autotune_process
            from .codegen.catlass.catlass_kernel import CATLASSTemplateCaller

            # only benchmark triton kernel in sub process for now.
            # ATen/Extern/Catlass kernel are still benchmarked in the current process.
            extern = [
                c
                for c in choices
                if isinstance(c, (ExternKernelCaller, CATLASSTemplateCaller))
            ]
            triton = [c for c in choices if c not in extern]

            timings = benchmark_in_current_process(extern)
            timings.update(autotune_process.benchmark_in_sub_process(triton))  # type: ignore[arg-type]
            return timings

        from .config import catlass as catlass_config

        if catlass_config.catlass_bench_use_profiling:
            benchmark = profiling_choices_in_current_process
        else:
            benchmark = (
                benchmark_in_sub_process
                if config.autotune_in_subproc
                else benchmark_in_current_process
            )

        return benchmark

    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    AlgorithmSelectorCache.__call__ = __call__
    AlgorithmSelectorCache.make_benchmark_fn = make_benchmark_fn
