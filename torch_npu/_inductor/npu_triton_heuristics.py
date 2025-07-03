# This file is based on triton_heuristics with heuristics designed for NPU
import copy
import functools
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from itertools import count
from typing import Any, Callable, List, Optional
import torch
from torch._logging import warning_once
import triton
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch._inductor.compile_fx import clone_preserve_strides
from torch._inductor.runtime.autotune_cache import AutotuneCache
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.runtime.runtime_utils import (
    create_bandwidth_info_str,
    get_num_bytes,

)
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    HeuristicType,
    unique_configs,
    hash_configs,
    Config,
    ASTSource,
    _find_names,
    get_first_attr,
    collected_calls,
    _dump_launch_params,
    builtins
)
from triton.compiler import CompiledKernel

try:
    from triton.backends.compiler import GPUTarget
    from triton.runtime.autotuner import OutOfResources
    import torch.autograd.profiler as autograd_profiler
except ImportError:
    GPUTarget = None
    OutOfResources = None
    autograd_profiler = None

from torch_npu.utils._error_code import ErrCode, pta_error

from .codegen.split_tiling import SplitTiling
from .utils import get_current_raw_stream
from .codegen.tile_generator import TileGenerator
from .codegen.triton_utils import get_aligned_numel
from .config import aggresive_autotune
from .config import log
from . import config as npu_config

kernel_idx = count()


class NPUCachingAutotuner(CachingAutotuner):
    def __init__(
            self,
            fn, 
            triton_meta,  # passed directly to triton
            configs,
            save_cache_hook,
            mutated_arg_names: List[str],  # see [Note: clone mutated buffers]
            optimize_mem,
            heuristic_type,
            size_hints=None,
            inductor_meta=None,  # metadata not relevant to triton
            custom_kernel=False,  # whether the kernel is inductor-generated or custom
            filename: Optional[str] = None,
            reset_to_zero_arg_names: Optional[List[str]] = None,
    ):
        super().__init__(fn, triton_meta, configs, save_cache_hook, mutated_arg_names, optimize_mem, heuristic_type,
                         size_hints, inductor_meta, custom_kernel, filename, reset_to_zero_arg_names)

        self.exceptions = []
        self.fn_name = None

    @staticmethod
    def api_accuracy_checker(expected, actual, kernel_name, dump_path):
        from msprobe.core.common.const import CompareConst
        from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import BENCHMARK_COMPARE_SUPPORT_LIST
        from msprobe.pytorch.api_accuracy_checker.triton_adapter.get_compare_result import get_compare_result
        from msprobe.pytorch.api_accuracy_checker.triton_adapter.precision_compare import precision_compare
        from msprobe.pytorch.api_accuracy_checker.triton_adapter.common.compare_utils import \
            convert_compare_column_to_row, print_check_details
        from msprobe.pytorch.api_accuracy_checker.triton_adapter.precision_standard.triton_standard_register import \
            exist_in_precision_standard

        dtype = actual.dtype

        # only float use precision standard
        if exist_in_precision_standard(kernel_name):
            if str(dtype) in BENCHMARK_COMPARE_SUPPORT_LIST:
                compare_column = precision_compare(kernel_name, expected, actual, dtype)  # calc metrics
                compare_row = convert_compare_column_to_row(compare_column, kernel_name)
                status = get_compare_result(compare_row, kernel_name)  # get compare results
                if status == CompareConst.ERROR:
                    log.warning(f'CHECK ACCURACY FAILED! kernel: {kernel_name}, Dump Path: {dump_path}')
                    print_check_details(compare_column, kernel_name)
                    actual.copy_(expected)
                checked_by_msprobe = True
            else:
                log.warning(f'The data type {dtype} is not supported for new precision standard. '
                            f'Check accuracy by tolerance method.')
                checked_by_msprobe = False
        else:
            log.warning(f'kernel_name {kernel_name} does not in new precision standard. '
                        f'Check accuracy by tolerance method.')
            checked_by_msprobe = False
        return checked_by_msprobe

    def precompile(self, warm_cache_only=False):
        # xpu_graph changed TORCHINDUCTOR_CACHE_DIR.
        # When TORCHINDUCTOR_COMPILE_THREADS > 1, multiprocessing's fork method
        # does not propagate TORCHINDUCTOR_CACHE_DIR into the child threads.
        # However, after all the child threads finished, the main thread reaches
        # here and inherits xpu_graph's TORCHINDUCTOR_CACHE_DIR. Then the main
        # thread finds the cache dir does not have any compiled kernel. It will
        # compile all kernels one by one.
        # So we directly replace TORCHINDUCTOR_CACHE_DIR with the standard cache dir.
        if ("xpu_graph" in os.getenv("TORCHINDUCTOR_CACHE_DIR", "")):
            import getpass
            import tempfile
            sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
            cache_dir = os.path.join(
                tempfile.gettempdir(),
                "torchinductor_" + sanitized_username,
            )
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton", "0")
        with self.lock:
            if self.launchers:
                return
            self.launchers = []
            compiled_binaries = []
            if not self.configs:
                raise RuntimeError("No triton configs are available")
            for c in self.configs:
                try:
                    compiled_binary, launcher = self._precompile_config(
                        c, warm_cache_only
                    )
                except Exception as e:
                    log.debug(
                        f"[thread {os.getpid()}][InductorNPU.precompile] Exception = {e}, kernel = {self.fn.__name__} config = {c}")
                    # Skip the config if the compilation fails
                    continue
                if launcher is not None:
                    self.launchers.append(launcher)
                    compiled_binaries.append(compiled_binary)

            if len(self.launchers) == 0:
                raise RuntimeError(
                    "No valid triton configs. Report a fatal compilation error"
                )

            self.configs = None

    def _precompile_config(self, cfg: Config, warm_cache_only: bool):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)

        for k, v in cfg.kwargs.items():
            if k not in self.fn.arg_names:
                continue
            compile_meta["constants"][k] = v

        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages

        compile_meta["debug"] = (
                os.getenv("INDUCTOR_ASCEND_DEBUG", 'false').lower() in ('true', '1') and
                config.assert_indirect_indexing and torch.version.hip is None
        )

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if ASTSource:
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                ),
            )

            cc_str = str(compile_meta["cc"])
            if "gfx10" in cc_str or "gfx11" in cc_str:
                rocm_warp_size = 32
            else:
                rocm_warp_size = 64

            if GPUTarget:
                target = GPUTarget(
                    compile_meta["device_type"],
                    compile_meta["cc"],
                    rocm_warp_size if torch.version.hip else 32,
                )
            else:
                target = (
                    (compile_meta["device_type"], compile_meta["cc"])
                    if not torch.version.hip
                    else [
                        compile_meta["device_type"],
                        compile_meta["cc"],
                        rocm_warp_size,
                    ]
                )

            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            if self.device_props.type == "hip":
                if "waves_per_eu" in compile_meta:
                    options["waves_per_eu"] = compile_meta["waves_per_eu"]
                if "matrix_instr_nonkdim" in compile_meta:
                    options["matrix_instr_nonkdim"] = compile_meta[
                        "matrix_instr_nonkdim"
                    ]
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn,)
            compile_kwargs = compile_meta
        if warm_cache_only:
            return (
                triton.compile(*compile_args, **compile_kwargs),
                None,
            )

        # importing from torch is safe now that precompile has returned
        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, compile_meta["device"]):  # type: ignore[attr-defined]
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())

            try:

                binary = triton.compile(*compile_args, **compile_kwargs)
                binary._init_handles()

            except Exception:
                log.exception(
                    "Triton compilation failed: %s\n%s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    compile_meta,
                )
                raise

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": CompiledKernel.launch_enter_hook,
            "launch_exit_hook": CompiledKernel.launch_exit_hook,
            "metadata": binary.packed_metadata
            if hasattr(binary, "packed_metadata")
            else binary.metadata,
            "shared": binary_shared,
        }

        scope["num_warps"] = (
            binary.num_warps
            if hasattr(binary, "num_warps")
            else binary.metadata.num_warps
        )

        scope["cta_args"] = (
            (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
            if hasattr(binary, "num_ctas")
            else (
                (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                if hasattr(binary, "metadata")
                else ()
            )
        )

        scope["function"] = get_first_attr(binary, "function", "cu_function")

        def get_launch_args_without_kernel_launch_metadata(
                input_grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                input_bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
        ):
            """
            Construct launch args before CompiledKernel.launch_metadata is added.
            """
            return (
                grid_0,
                grid_1,
                grid_2,
                num_warps,
                *cta_args,
                shared,
                stream,
                function,
                launch_enter_hook,
                launch_exit_hook,
                metadata,
            )

        # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
        # `bin.launch_metadata` is relatively expensive, and returns None unless a
        # `launch_enter_hook` is installed.  So if we don't have that hook installed,
        # we want to burn None in to the launch args with zero overhead.
        if binary.launch_enter_hook:

            def get_launch_args_with_kernel_launch_metadata(
                    input_grid,
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    input_bin,
                    launch_enter_hook,
                    launch_exit_hook,
                    num_warps,
                    shared,
                    cta_args,
                    args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    input_bin.launch_metadata(input_grid, stream, *args),
                    launch_enter_hook,
                    launch_exit_hook,
                )

        else:

            def get_launch_args_with_kernel_launch_metadata(
                    input_grid,
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    input_bin,
                    launch_enter_hook,
                    launch_exit_hook,
                    num_warps,
                    shared,
                    cta_args,
                    args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    None,
                    launch_enter_hook,
                    launch_exit_hook,
                )

        scope["get_launch_args"] = (
            get_launch_args_with_kernel_launch_metadata
            if hasattr(binary, "launch_metadata")
            else get_launch_args_without_kernel_launch_metadata
        )

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")

        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                args = {', '.join(call_args)},
                launch_args = get_launch_args(
                    grid, grid_0, grid_1, grid_2, stream, function,
                    metadata, bin, launch_enter_hook, launch_exit_hook,
                    num_warps, shared, cta_args, args
                )
                runner(*launch_args, *args)
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

    def save_gpu_kernel(self, input_grid, input_stream, input_launcher):
        self.save_npu_kernel(input_grid, input_stream, input_launcher)

    def save_npu_kernel(self, input_grid, input_stream, input_launcher):
        if callable(input_grid):
            grid_x, grid_y, grid_z = input_grid(input_launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = input_grid

        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name

        if key is None:
            raise RuntimeError("assert key is not None, kernel_name can not be None")
        params = {
            "mangled_name": (
                input_launcher.bin.metadata.name
                if hasattr(input_launcher.bin.metadata, "name")
                else input_launcher.bin.metadata["name"]
            ),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "num_warps": (
                input_launcher.bin.num_warps
                if hasattr(input_launcher.bin, "num_warps")
                else input_launcher.bin.metadata.num_warps
            ),
            "shared_mem": (
                input_launcher.bin.shared
                if hasattr(input_launcher.bin, "shared")
                else input_launcher.bin.metadata.shared
            ),
            "stream": input_stream,
            # User defined triton kernels will have arbitrary kwarg names
            "meta": input_launcher.config.kwargs,
        }
        from torch._inductor.codecache import CudaKernelParamCache

        bin_type = "npubin"
        binary = input_launcher.bin.asm[bin_type]  # npubin type = npubin
        CudaKernelParamCache.set(key, params, binary, bin_type='cubin')  # CudaKernelParam

        self.cuda_kernel_saved = True

    # bench method is called by torch, grid can not be modified
    def bench(self, launcher, *args, grid, with_profiler=False, **kwargs):
        """Measure the performance of a given launcher"""

        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
                "spill_threshold", 16
        ):
            return float("inf")

        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        def kernel_call():
            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                grid=grid,
                stream=stream,
            )

        if with_profiler:
            from torch._inductor.utils import do_bench_using_profiling
            ret = do_bench_using_profiling(kernel_call, warmup=10, rep=1)

        # remove fast_flush=True for high version triton
        ret = benchmarker.benchmark_gpu(kernel_call, rep=1)
        return ret

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        start_time = time.time_ns()
        timings = self.benchmark_all_configs(*args, **kwargs)
        benchmark_time_taken_ns = time.time_ns() - start_time
        self.launchers = [builtins.min(timings, key=timings.get)]
        self.autotune_time_taken_ns = (
                self.precompile_time_taken_ns + benchmark_time_taken_ns
        )
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config, self.autotune_time_taken_ns)

    def get_fx_graph_call(self, auto_fallback=False):
        kernel_name = self.inductor_meta.get("kernel_name", "triton_")
        traced_graph_hash = self.inductor_meta.get("traced_graph_hash")
        dump_dir = self.inductor_meta.get("traced_graph_dir", "")
        dump_path = os.path.join(dump_dir, traced_graph_hash)
        if dump_dir == "" or not os.path.exists(dump_path):
            return None, None, None, None
        sys.path.append(dump_path)
        fx_module = importlib.import_module(traced_graph_hash)
        sys.path.remove(dump_path)

        model = fx_module.model
        num_inputs = fx_module.num_inputs
        num_outputs = fx_module.num_outputs
        non_contiguous_indices = fx_module.non_contiguous_indices
        mismatch_indices_shapes = fx_module.mismatch_indices_shapes

        def fx_graph_call(*fx_args):
            fx_inputs = [fx_args[idx].contiguous() if idx in non_contiguous_indices['inputs'] else \
                             fx_args[idx] for idx in range(num_inputs)]
            if len(mismatch_indices_shapes):
                for ind, shape in mismatch_indices_shapes.items():
                    if ind >= num_inputs:
                        break
                    fx_inputs[ind] = fx_inputs[ind].reshape(shape)
            model_outputs = model.forward(*fx_inputs)
            for idx, (out1, out2) in enumerate(zip(model_outputs, fx_args[num_inputs:(num_inputs + num_outputs)])):
                out1 = out1.reshape(out2.shape)
                if idx in non_contiguous_indices['outputs']:
                    out2.copy_(out1)
                else:
                    out2.data = out1.data

        def fallback_call(*args):
            fx_args = [args[idx] for idx in fx_module.call_args_mapping]
            return fx_graph_call(*fx_args)

        if auto_fallback:
            return fallback_call, kernel_name, None, None
        return fx_graph_call, kernel_name, dump_path, fx_module

    def data_dump(self, *args, dump_path=None):
        data_dump_path = os.path.join(dump_path, 'data.pth')
        torch.save(args, data_dump_path)

    def get_fn_name(self):
        if self.fn_name is not None:
            return self.fn_name
        try:
            self.fn_name = self.fn.fn.__name__
        except AttributeError:
            self.fn_name = "unknown"
        return self.fn_name

    def fallback_to_fx(self, *args, launcher, grid_, stream, **kwargs):
        """
        Try to fallback kernel to fx graph call according to kernel id.
        """
        def should_fallback():
            fallback_id = npu_config.force_fallback_kernel_id
            if fallback_id != "all" and not isinstance(fallback_id, list):
                raise RuntimeError("torch_npu._inductor.config.aot_inductor.force_fallback_kernel_id "
                                   "should be set to 'all' or List, e.g, [1, 2, 10]." + pta_error(ErrCode.VALUE))
        
        if not should_fallback():
            return None
        
        fx_graph_call, _, _, fx_module = self.get_fx_graph_call()
        if not fx_graph_call:
            return None

        call_outputs_indices = fx_module.call_args_mapping[fx_module.num_inputs:]
        fx_args = []
        for idx in fx_module.call_args_mapping:
            arg = args[idx]
            if isinstance(arg, torch.Tensor):
                fx_arg = clone_preserve_strides(arg).float() if arg.dtype == torch.bfloat16 else clone_preserve_strides(
                    arg)
                fx_args.append(fx_arg)

        fx_graph_call(*fx_args)
        for actual, expected in zip([args[i] for i in call_outputs_indices], fx_args[fx_module.num_inputs:]):
            if actual.dtype != expected.dtype:
                expected = expected.to(actual.dtype)
            actual.copy_(expected)
        for arg in fx_args:
            del arg
        return True
        

    def check_accuracy(self, *args, launcher, grid, stream, **kwargs):
        fx_graph_call, kernel_name, dump_path, fx_module = self.get_fx_graph_call()
        if not fx_graph_call:
            return None
        call_outputs_indices = fx_module.call_args_mapping[fx_module.num_inputs:]
        self.data_dump(*args, dump_path=dump_path)

        fx_args = []
        for idx in fx_module.call_args_mapping:
            arg = args[idx]
            if isinstance(arg, torch.Tensor):
                fx_arg = clone_preserve_strides(arg).float() if arg.dtype == torch.bfloat16 else clone_preserve_strides(
                    arg)
                fx_args.append(fx_arg)

        fx_graph_call(*fx_args)

        ret = launcher(
            *args,
            **kwargs,
            grid=grid,
            stream=stream,
        )

        try:
            import msprobe
            has_msprobe = True
        except ImportError:
            has_msprobe = False
            warning_once(log, "msprobe import failed, please check. "
                              "It may be due to missing dependencies or other factors. "
                              "Check accuracy by tolerance method.")
        for actual, expected in zip([args[i] for i in call_outputs_indices], fx_args[fx_module.num_inputs:]):
            if actual.dtype != expected.dtype:
                expected = expected.to(actual.dtype)
            checked_by_msprobe = False
            if has_msprobe:
                checked_by_msprobe = self.api_accuracy_checker(expected, actual, kernel_name, dump_path)
            if not has_msprobe or not checked_by_msprobe:
                acc_comp_tol = npu_config.acc_comp_tol.get(actual.dtype, npu_config.acc_comp_tol['default'])
                rtol = acc_comp_tol['rtol']
                atol = acc_comp_tol['atol']

                matches = torch.isclose(
                    actual, expected, rtol=rtol, atol=atol, equal_nan=False
                )
                if not matches.all():
                    abs_diff = torch.abs(actual - expected)
                    rel_diff = abs_diff / torch.abs(expected)
                    rel_diff.masked_fill_(matches, 0)
                    log.warning(f"CHECK ACCURACY FAILED! Greatest Relative Difference: {rel_diff.max().item()}, "
                                f"Kernel Name: {kernel_name}, Dump Path: {dump_path}")
                    actual.copy_(expected)
                del matches
        for arg in fx_args:
            del arg
        return True

    def debug_kernel_in_run(self, *args, launcher, grid_, stream, **kwargs):
        '''
        Save tensors for kernel args and outputs before and after kernel execute.
        These tensors can be load and compared with tensors dumped by aot-inductor cpp runtime.
        '''
        dump_path = npu_config.aot_inductor.dump_path_py
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)

        idx = next(kernel_idx)
        fn_name = self.get_fn_name()
        dump_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        torch.npu.synchronize()
        torch.save(dump_args, f"{dump_path}/{idx}_{fn_name}_before.pt")

        result = super().run(*args, grid=grid_, stream=stream, **kwargs)

        torch.npu.synchronize()
        torch.save(dump_args, f"{dump_path}/{idx}_{fn_name}_after.pt")
        return result


    def run(
            self, *args, grid, stream, benchmark_run=False, **kwargs
    ):  # type:ignore[override]
        if self.triton_interpret:
            return self.fn[grid](
                *args,
                **kwargs,
                **self.configs[0].kwargs,
            )

        if hasattr(self.launchers[0], "fallback"):
            return self.launchers[0](
                *args,
                **kwargs,
            )

        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                start_time = time.time_ns()
                self.precompile()
                self.precompile_time_taken_ns = time.time_ns() - start_time
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid, **kwargs)

        if not getattr(
                self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            self.launchers = [
                self.coordinate_descent_tuning(
                    self.launchers[0], *args, grid=grid, **kwargs
                )
            ]

        (launcher,) = self.launchers
        if launcher.store_cubin and (not benchmark_run or not self.cuda_kernel_saved):
            self.save_gpu_kernel(grid, stream, launcher)

        if self.dump_launch_params:
            _dump_launch_params(args, kwargs, launcher, self.fn.__name__)

        if npu_config.check_accuracy:
            if self.check_accuracy(*args, launcher=launcher, grid=grid, stream=stream, **kwargs):
                return
        elif npu_config.force_fallback_kernel_id:
            fallback_result = self.fallback_to_fx(*args, launcher=launcher, grid_=grid, stream=stream, **kwargs)
            if fallback_result is not None:
                log.debug(f"fallback kernel {self.get_fn_name()} to fx graph call.")
                return
            else:
                log.warning(f"kernel {self.get_fn_name()} could not fallback to fx.")
        elif npu_config.aot_inductor.debug_kernel_in_run:
            return self.debug_kernel_in_run(*args, launcher=launcher, grid_=grid, stream=stream, **kwargs)

        # it is faster than entering and exiting a context manager, even if the context
        # manager is a nullcontext.
        if autograd_profiler._is_profiler_enabled:
            # grid can be a tuple of ints or a string.
            if isinstance(grid, tuple):
                grid_info = str(grid)
            else:
                grid_info = getattr(grid, "grid_fn_str", "")

            with torch._C._profiler._RecordFunctionFast(
                    self.inductor_meta.get("kernel_name", "triton kernel"),
                    args,
                    {
                        "kernel_file": (self.filename or ""),
                        "kernel_hash": self.kernel_hash,
                        "kernel_backend": "triton",
                        "grid": grid_info,
                        "stream": stream,
                    },
            ):
                return launcher(
                    *args,
                    **kwargs,
                    grid=grid,
                    stream=stream,
                )
        else:
            return launcher(
                *args,
                **kwargs,
                grid=grid,
                stream=stream,
            )


class NPUDebugAutotuner(NPUCachingAutotuner):
    def __init__(self, *args, regex_filter="", **kwargs):
        self.regex_filter = regex_filter
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, input_grid, stream):
        possible_names = _find_names(self)
        kernel_name = f"{max(possible_names, key=len)}"
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=input_grid, stream=stream)
        (launcher,) = self.launchers

        if self.cached is None:
            ms = self.bench(launcher, *args, input_grid=input_grid)
            num_in_out_ptrs = len(
                [
                    arg_name
                    for arg_name in self.fn.arg_names
                    if arg_name.startswith("in_out_ptr")
                ]
            )
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
            gb_per_s = num_gb / (ms / 1e3)
            self.cached = (ms, num_gb, gb_per_s, kernel_name)
        else:
            ms, num_gb, gb_per_s, kernel_name = self.cached
        collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
        print(
            create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}")
        )


def cached_autotune(
        size_hints: Optional[List[int]],
        configs: List[Config],
        triton_meta,
        heuristic_type,
        filename=None,
        inductor_meta=None,
        custom_kernel=False,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    if not (len(configs) == 1 or filename):
        raise RuntimeError("assert len(configs) == 1 or filename")

    inductor_meta = {} if inductor_meta is None else inductor_meta

    disabled = inductor_meta.get("force_disable_caches", False)

    # on disk caching logic and/or remote caching
    autotune_cache = None
    if (
            not disabled
            and filename is not None
            and (len(configs) > 1 or inductor_meta.get("coordinate_descent_tuning"))
            and not os.environ.get("TRITON_INTERPRET", "0") == "1"
    ):
        configs_hash = hash_configs(configs)

        autotune_cache = AutotuneCache.create(inductor_meta, filename, configs_hash)
        if autotune_cache:
            best_config = autotune_cache.read_best(inductor_meta, configs)
            if best_config:
                configs = [best_config]
    else:
        if disabled:
            log.debug("autotune caching is disabled by config.force_disable_caches")

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())
    optimize_mem = inductor_meta.pop("optimize_mem", True)

    if "restore_value" in triton_meta:
        mutated_arg_names += triton_meta.pop("restore_value")

    reset_to_zero_arg_names: List[str] = []
    if "reset_to_zero" in triton_meta:
        reset_to_zero_arg_names.extend(triton_meta.pop("reset_to_zero"))

    def decorator(fn):

        if inductor_meta.get("profile_bandwidth"):
            return NPUDebugAutotuner(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=inductor_meta["profile_bandwidth_regex"],
                with_profiler=inductor_meta[
                    "profile_bandwidth_with_do_bench_using_profiling"
                ],
                configs=configs,
                save_cache_hook=autotune_cache and autotune_cache.save,
                mutated_arg_names=mutated_arg_names,
                reset_to_zero_arg_names=reset_to_zero_arg_names,
                optimize_mem=optimize_mem,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
                filename=filename,
                with_bandwidth_info=True,
            )
        return NPUCachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=autotune_cache and autotune_cache.save,
            mutated_arg_names=mutated_arg_names,
            reset_to_zero_arg_names=reset_to_zero_arg_names,
            optimize_mem=optimize_mem,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
            filename=filename,
        )

    return decorator


######################################################
## Main entry points for triton kernel invocation   ##
## adapts original heuristics for NPU arch, and     ##
## redirect to NPUCaching autotuner                 ##
######################################################

def grid(*numels):
    def grid_fn(meta):
        split_axis = meta["split_axis"]
        split_blocks = meta["split_blocks"]
        programs = []
        for i, order in enumerate(split_axis):
            if not numels:
                continue
            numel = numels[order]
            block = split_blocks[i]
            programs.append((numel + block - 1) // block)

        for _ in range(3 - len(programs)):
            programs.append(1)
        return tuple(programs)

    return grid_fn


# split:sizeof split, xblock:axis1 length, rblock:axis2 length
def triton_config_npu_index(
        size_hints,
        inductor_meta,
        triton_meta=None,
        reduction=False,
        persistent_reduction=False,

) -> List[Config]:
    num_warps = 1
    num_stages = 1
    configs = []
    split_axis = inductor_meta["split_axis"]
    tiling_axis = inductor_meta["tiling_axis"]
    low_dims = inductor_meta["low_dims"]
    split_axis_dtype = inductor_meta["split_axis_dtype"]
    axis_names = inductor_meta["axis_names"]
    dual_reduction = inductor_meta["dual_reduction"]

    tile_generator = TileGenerator(size_hints, axis_names, tiling_axis, split_axis, low_dims,
                                   persistent_reduction=persistent_reduction, configs=configs,
                                   dtype=split_axis_dtype, dual_reduction=dual_reduction)

    tile_generator.descend_split_tiling()

    if not configs:
        cfg = {}
        for x in split_axis:
            cfg[f"{axis_names[x].upper()}BLOCK"] = size_hints[x]
        if not cfg:
            cfg["dummy"] = 1
        tmp = Config(cfg, num_warps=num_warps, num_stages=num_stages)
        configs.append(tmp)

    for cfg in configs:
        split_blocks = [None for x in split_axis]
        for i, axis in enumerate(split_axis):
            name = axis_names[axis]
            block_name = f"{name.upper()}BLOCK"
            split_blocks[i] = cfg.kwargs[block_name]
        cfg.kwargs["split_axis"] = tuple(split_axis)
        cfg.kwargs["split_blocks"] = tuple(split_blocks)

    return configs


def pointwise_npu_index(
        size_hints,
        triton_meta,
        tile_hint=None,
        filename=None,
        min_elem_per_thread=0,
        inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    triton_config_with_settings = functools.partial(
        triton_config_npu_index
    )
    return cached_autotune(
        size_hints,
        triton_config_with_settings(size_hints, inductor_meta=inductor_meta),
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )


def reduction_npu_index(
        size_hints,
        reduction_hint=False,
        triton_meta=None,
        filename=None,
        inductor_meta=None,
):
    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if triton_meta is None:
        raise RuntimeError("assert triton_meta is not None")

    contiguous_config = triton_config_npu_index(size_hints, inductor_meta=inductor_meta, reduction=True)
    return cached_autotune(
        size_hints,
        [
            *contiguous_config,
        ],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.REDUCTION,
    )


def persistent_reduction_npu_index(
        size_hints,
        reduction_hint=False,
        triton_meta=None,
        filename=None,
        inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    configs = triton_config_npu_index(size_hints, inductor_meta=inductor_meta, reduction=True,
                                      persistent_reduction=True)

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


@dynamo_timed
def benchmark_all_configs(self, *args, input_grid, **kwargs):
    print(f"candidate launcher count = {len(self.launchers)}")

    tilling_kernel_list = []

    def kernel_call(launcher):
        def call_kernel():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
                )
            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                grid=input_grid,
                stream=stream,
            )

        return call_kernel

    for launcher in self.launchers:
        if not self.custom_kernel and launcher.n_spills > config.triton.spill_threshold:
            return float("inf")

        stream = self.gpu_device.get_raw_stream(  # type: ignore[call-arg]
            self.gpu_device.current_device()
        )
        tilling_kernel_list.append(kernel_call(launcher))

    def do_batch_benchmark(tilling_kernel_list):

        def delete_file(base_path):
            import shutil
            if os.path.exists(base_path):
                shutil.rmtree(base_path)

        import torch_npu

        stream = torch.npu.current_stream()
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False
        )

        import uuid
        random_uuid = uuid.uuid4().hex
        md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()

        from torch_npu._inductor.config import profile_path

        torch_path = profile_path + md5_hash
        rep = 1
        with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.NPU
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=rep, repeat=1, skip_first=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config) as prof:
            stream.synchronize()
            for _ in range(rep + 3):
                for fn in tilling_kernel_list:
                    fn()
                prof.step()
            stream.synchronize()

        import pandas as pd
        for root, _, files in os.walk(torch_path):
            for file in files:
                if file != 'kernel_details.csv':
                    continue
                target_file = os.path.join(root, file)
                df = pd.read_csv(target_file)
                triton_rows = df[df['Name'].str.startswith('triton', na=False)]
                ret = triton_rows['Duration(us)'].astype(float).tolist()
                delete_file(torch_path)
                return ret

        delete_file(torch_path)
        return []

    try:
        timinglist = do_batch_benchmark(tilling_kernel_list)
        if not len(timinglist) == len(self.launchers):
            raise RuntimeError("not len(timinglist) == len(self.launchers)")
        timings = {launcher: timing for launcher, timing in zip(self.launchers, timinglist)}
    except Exception as e:
        print("some cases in batch benchmark has error! Logging Exception as:")
        print(e)
        print("switched to single bench...")
        timings = {
            launcher: self.bench(launcher, *args, **kwargs)
            for launcher in self.launchers
        }

    for k, v in timings.items():
        self.coordesc_tuner.cache_benchmark_result(k.config, v)

    if log.isEnabledFor(logging.DEBUG):
        for k, v in timings.items():
            log.debug(
                "%s: %f, nreg %d, nspill %d, #shared-mem %s",
                k.config,
                v,
                k.n_regs,
                k.n_spills,
                k.shared,
            )
    return timings
