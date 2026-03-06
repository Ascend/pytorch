import builtins
import contextlib
import dataclasses
import functools
import inspect
import itertools
import json
import logging
import math
import operator
import os
import sys
import textwrap
import time
from collections import namedtuple
from concurrent.futures import as_completed, ThreadPoolExecutor, Future
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from unittest.mock import patch

import sympy
from filelock import FileLock

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.utils import counters, dynamo_timed, identity, preserve_rng_state
from torch._inductor import config, ir
from torch._inductor.ir import ChoiceCaller
from torch._inductor.utils import restore_stdout_stderr, sympy_product, unique, Placeholder
from torch._inductor.virtualized import V
from torch._inductor.codegen.triton import (
    texpr,
    TritonScheduling,
    gen_common_triton_imports,
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
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.exc import CppCompileError
from torch.utils._ordered_set import OrderedSet
from torch_npu._inductor.codegen.triton import gen_triton_ext_imports

from ..profiler import tensorboard_trace_handler


log = logging.getLogger("torch._inductor")


class NPUCompileError(CppCompileError):
    pass


class NPUTritonTemplate(TritonTemplate):
    """NPU-specific Triton template for kernel generation.
    
    This class extends TritonTemplate to provide NPU-specific optimizations
    and configurations for Triton kernel generation.
    """
    
    index_counter = itertools.count()

    def __init__(self, name: str, grid: Any, source: str, debug: bool = False) -> None:
        """Initialize NPU Triton template.
        
        Args:
            name: Template name for identification
            grid: Grid function for kernel launch configuration
            source: Triton kernel source code
            debug: Enable debug mode for verbose output
        """
        super().__init__(name, grid, source, debug)

    def generate(
        self,
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        num_stages: int,
        num_warps: int,
        prefix_args: int = 0,
        suffix_args: int = 0,
        epilogue_fn: Callable = identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        mutated_inputs: Optional[list[ir.IRNode]] = None,
        call_sizes: Optional[list[sympy.Expr]] = None,
        workspace_arg: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[ir.ChoiceCaller]:
        defines = StringIO()
        kwargs["ALLOW_TF32"] = "False"
        for name, val in kwargs.items():
            defines.write(f"{name} : tl.constexpr = {val}\n")
        defines = defines.getvalue()

        fake_out = ir.Buffer(name="buf_out", layout=layout)
        kernel_name = f"triton_{self.name}"

        numel = sympy_product(layout.size)
        buffers = itertools.chain(input_nodes, (fake_out,))
        if not TritonScheduling.can_use_32bit_indexing(numel, buffers):
            raise NotImplementedError(
                "64-bit indexing is not yet implemented for triton templates"
            )

        if call_sizes is None:
            call_sizes = layout.size

        kernel_options = {
            "input_nodes": input_nodes,
            "defines": defines,
            "num_stages": num_stages,
            "num_warps": num_warps,
            "grid_fn": self.grid,
            "meta": kwargs,
            "call_sizes": call_sizes,
            "prefix_args": prefix_args,
            "suffix_args": suffix_args,
            "epilogue_fn": epilogue_fn,
            "subgraphs": subgraphs,
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
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, tuple(kernel.args.sizevars.keys())),
            fallback=config.unbacked_symint_fallback,
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
            *V.graph.sizevars.size_hints(
                call_sizes,
                fallback=config.unbacked_symint_fallback,
            ),
            kwargs,
        )
        bmreq_cls: type[TritonBenchmarkRequest]
        if layout.device.type == "cpu":
            bmreq_cls = TritonCPUBenchmarkRequest
        else:
            bmreq_cls = TritonGPUBenchmarkRequest
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
            workspace_arg=workspace_arg,
        )

        return TritonTemplateCaller(
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
        use_jit: bool = False,
        prefix_args: int = 0,
        suffix_args: int = 0,
        epilogue_fn: Callable = identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[Any] = None,
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
            kernel_name,
            input_nodes,
            output_node,
            defines,
            num_stages,
            num_warps,
            grid_fn,
            meta,
            call_sizes,
            use_jit,
            prefix_args,
            suffix_args,
            epilogue_fn,
            subgraphs,
            workspace_arg,
        )

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
            code.splice(gen_common_triton_imports())
            code.splice(gen_triton_ext_imports())
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
        return "<DEF_KERNEL>"


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
    ) -> Any:
        from .codegen.catlass.catlass_kernel import CATLASSTemplateCaller

        # Templates selected with input_gen_fns require specific input data to avoid IMA
        # Passing custom input gen fns to benchmark_fusion NYI, so skip deferred template selection
        # TODO(jgong5): support multi-template on CPU
        if input_gen_fns is not None or layout.device.type == "cpu":
            return_multi_template = False

        choices = [choice for choice in choices if choice is not None]

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

        if len(choices) == 1:
            if not isinstance(choices[0], CATLASSTemplateCaller):
                # CATLASSTemplateCaller still needs to go through autotuning process to retrieve workspace size.
                return choices[0].output_node()

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

            if (
                sys.version_info.major == 3
                and sys.version_info.minor == 11
                and sys.version_info.micro <= 8
            ):
                return no_op

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
            if precompile_func := self.precompile_cache.get(precompile_key):
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
                        counters["inductor"]["select_algorithm_num_precompiles"] += 1
                        log.info(
                            "Precompiling benchmark choice %s took %.02fs",
                            futures[future],
                            elapsed_times[future],
                        )

                executor.shutdown(wait=True)

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

        precompile_fn = precompile(choices)

        if return_multi_template and (config.max_autotune or config.max_autotune_gemm):

            def get_timings():
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

            return torch._inductor.ir.TensorBox.create(
                torch._inductor.ir.MultiTemplateBuffer(
                    layout,
                    input_nodes,
                    get_timings,
                    choices,
                    allowed_prologue_inps,
                )
            )

        timings = do_autotuning(precompile_fn)
        if timings == {} or choices[0] not in timings:
            return choices[0].output_node()

        selected_key = builtins.min(timings, key=timings.__getitem__)
        selected_choice = selected_key.output_node()
        log.debug("selected choice: %s", str(selected_choice))
        return selected_choice

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
                        V.graph.sizevars.size_hints(
                            input_node.get_size(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                        V.graph.sizevars.size_hints(
                            input_node.get_stride(),
                            fallback=config.unbacked_symint_fallback,
                        ),
                        V.graph.sizevars.size_hint(
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
                out, out.size(), out.stride(), V.graph.sizevars.size_hint(layout.offset)
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
            log.debug(f"{len(choices)} tuning requests:")

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
            buffer.zero_()
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
                        buffer.zero_()
                        fn(*args, **kwargs)
                        torch.npu.synchronize()
                    # One aclnn op may be seperated into multiple ops, recorded in kernel_details.csv,
                    # which makes us hard to analyze the kernel_detail.csv. Therefore, an abs operation is added here,
                    # aimming to help us recognize different ops.
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
                    filter_cond = ~df["Name"].str.contains(r"zero|ZerosLike", case=False, na=False)
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
                    log.error(
                        "NPU compilation error during autotuning: \n%s. \nIgnoring this choice.",
                        str(e),
                    )
                    timing = float("inf")
                except NotImplementedError as e:
                    log.warning("Not yet implemented: %s", e)
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
                            log.warning(e)
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
                if isinstance(c, ExternKernelCaller) or isinstance(c, CATLASSTemplateCaller)
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
