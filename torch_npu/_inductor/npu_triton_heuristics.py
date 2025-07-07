# This file is based on triton_heuristics with heuristics designed for NPU
import copy
import functools
import hashlib
import importlib
import json
import logging
import dataclasses
import os
import re
import sys
import time
from itertools import count
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING, Union, List
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
from torch._inductor.utils import triton_version_uses_attrs_dict
from torch.utils._ordered_set import OrderedSet
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
    builtins,
    NoTritonConfigsError,
    TritonCompileResult,
    GridExpr,
    config_to_dict
)
from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key
from triton.compiler import CompiledKernel
from torch._inductor.triton_bundler import TritonBundler

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


@dataclasses.dataclass
class GridNpu(GridExpr):
    numels: List[str] = None

    def generate(self, meta: dict[str, int]) -> None:
        numel_args = []
        split_axis = meta.get("split_axis", None)
        split_blocks = meta.get("split_blocks", None)
        if split_axis is None or split_blocks is None:
            raise RuntimeError(f"Could not get split_axis or split_blocks from meta {meta}.")

        def grid_fn(i):
            if i >= len(split_axis):
                return "1"
            axis = split_axis[i]
            block = split_blocks[i]
            return f"({self.numels[axis]} + {block} - 1) // {block}"
        self.x_grid = grid_fn(0)
        self.y_grid = grid_fn(1)
        self.z_grid = grid_fn(2)


class GridExprNpu(GridExpr):
    @staticmethod
    def from_meta_and_set_numel(
        inductor_meta: dict[str, Any],
        cfg: Union[Config, dict[str, int]],
        numels: List[str],
        mode: Literal["python", "cpp"] = "python",
    ) -> GridExpr:
        grid_cls = globals()[inductor_meta["grid_type"]]
        if not issubclass(grid_cls, GridNpu):
            raise AssertionError(f"grid_type in inductor_meta must be subclass of GridNpu"
                                 f"but got {inductor_meta['grid_type']}")
        grid = grid_cls(inductor_meta=inductor_meta, mode=mode, numels=numels)
        if isinstance(cfg, Config):
            cfg = config_to_dict(cfg)
        grid.generate(cfg)
        return grid


class TritonCompileResultNpu(TritonCompileResult):
    def make_launcher(self):
        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn
        binary._init_handles()

        known_constants = OrderedSet(
            arg for i, arg in enumerate(fn.arg_names) if i in fn.constexprs
        )
        none_args = OrderedSet(
            k
            for k, v in compile_meta["constants"].items()
            if v is None and k not in known_constants
        )
        none_args = none_args.difference(OrderedSet(compile_meta["signature"].keys()))

        if triton_version_uses_attrs_dict():
            call_args = fn.arg_names
            def_args = fn.arg_names
            if (
                "num_warps" in compile_meta["constants"]
                or "num_stages" in compile_meta["constants"]
            ):
                # num_warps/num_stages are special implicit args that are not in the signature
                # see test_triton_kernel_special_params
                def_args = [
                    arg for arg in def_args if arg not in ("num_warps", "num_stages")
                ]
                repl = {
                    k: str(compile_meta["constants"].get(k))
                    for k in ("num_warps", "num_stages")
                }
                call_args = [repl.get(arg, arg) for arg in call_args]
        else:
            call_args = [
                arg
                for i, arg in enumerate(fn.arg_names)
                if i not in fn.constexprs and arg not in none_args
            ]
            cfg_dict = config_to_dict(cfg)
            def_args = [
                name
                for name in fn.arg_names
                if name not in cfg_dict and name not in none_args
            ]

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": binary.__class__.launch_enter_hook,
            "launch_exit_hook": binary.__class__.launch_exit_hook,
            "metadata": (
                binary.packed_metadata
                if hasattr(binary, "packed_metadata")
                else binary.metadata
            ),
            "shared": binary_shared,
            "num_warps": (
                binary.num_warps
                if hasattr(binary, "num_warps")
                else binary.metadata.num_warps
            ),
            "cta_args": (
                (
                    binary.num_ctas,
                    *get_first_attr(binary, "cluster_dims", "clusterDims"),
                )
                if hasattr(binary, "num_ctas")
                else (
                    (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                    if hasattr(binary, "metadata")
                    else ()
                )
            ),
            "function": get_first_attr(binary, "function", "cu_function"),
            "runner": get_first_attr(binary, "run", "c_wrapper"),
        }

        if not hasattr(binary, "launch_metadata"):
            # launch args before CompiledKernel.launch_metadata is added.
            # TODO(jansel): delete this branch in mid-2025
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "num_warps",
                "*cta_args",
                "shared",
                "stream",
                "function",
                "launch_enter_hook",
                "launch_exit_hook",
                "metadata",
                *call_args,
            ]
        else:
            if binary.__class__.launch_enter_hook:
                launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"
            else:
                launch_metadata = "None"
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "stream",
                "function",
                "metadata",
                launch_metadata,
                "launch_enter_hook",
                "launch_exit_hook",
                *call_args,
            ]

        if "extra_launcher_args" in self.inductor_meta:
            def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

        numels = [
            arg
            for arg in fn.arg_names
            if "_numel" in arg
        ]
        grid = GridExprNpu.from_meta_and_set_numel(self.inductor_meta, cfg, numels)
        # grid.prefix is usually empty, grid.x_grid is something like `-(xnumel//-1024)`
        lines = [
            f"def launcher({', '.join(def_args)}, stream):",
            *[f"    {line}" for line in grid.prefix],
            f"    grid_0 = {grid.x_grid}",
            f"    grid_1 = {grid.y_grid}",
            f"    grid_2 = {grid.z_grid}",
            f"    runner({', '.join(runner_args)})",
        ]
        exec("\n".join(lines), scope)

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
                # arg filtering wasn't done above
                cfg_dict = config_to_dict(cfg)
                def_args = [x for x in def_args if x not in cfg_dict]
                call_args = [
                    x
                    for x in call_args
                    if compile_meta["signature"].get(x, "constexpr") != "constexpr"
                    and x not in none_args
                ]
            launcher.def_args = def_args
            launcher.call_args = call_args
        return launcher


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

    def precompile(
        self,
        warm_cache_only=False,
        reload_kernel: Optional[Callable[[], CachingAutotuner]] = None,
    ):
        if warm_cache_only:
            self._precompile_worker()
            return
        with self.lock:
            # Helper function for reloading a kernel generated in a worker
            # in the parent class. Normally we don't need to reload the kernel
            # in the parent process, but in certain cases (coordesc tuning, dynamic_scale_rblock),
            # we need to actually run compilation on the parent process
            if reload_kernel is not None:
                self._reload_kernel = reload_kernel
            self._precompile_worker()
            self._make_launchers()

    def _precompile_worker(self):
        if self.compile_results:
            for result in self.compile_results:
                TritonBundler.put(
                    triton_hash_to_path_key(result.kernel.hash),
                    self.triton_meta.get("device", 0),
                )
            return
        if self.launchers:
            raise AssertionError("Before _precompile_worker, launchers must bt empty")

        if not self.configs:
            raise NoTritonConfigsError("No triton configs are available")

        compile_results = []
        exc = None
        exc_stack = ""
        for c in self.configs:
            try:
                compile_results.append(self._precompile_config(c))
            except Exception as e:
                import traceback
                exc_stack = traceback.format_exc()
                exc = e
        if len(compile_results) == 0:
            raise NoTritonConfigsError(
                f"No valid triton configs. {type(exc).__name__}: {exc} \nStack trace:{exc_stack}"
            )
        self.compile_results = compile_results
        self.configs = None

    def _precompile_config(self, cfg: Config) -> TritonCompileResultNpu:
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        cfg_kwargs = cfg.kwargs
        for k, v in cfg_kwargs.items():
            if k not in self.fn.arg_names:
                continue
            compile_meta["constants"][k] = v

        for i in self.fn.constexprs:
            arg_name = self.fn.arg_names[i]
            if arg_name not in compile_meta["constants"] and (
                arg_name == "num_warps" or arg_name == "num_stages"
            ):
                compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = (
            os.getenv("INDUCTOR_ASCEND_DEBUG", 'false').lower() in ('true', '1') 
            and self.inductor_meta.get("assert_indirect_indexing", True)
            and not self.inductor_meta.get("is_hip", False)
        )

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if not ASTSource:
            raise RuntimeError("Installed triton version too old, please upgrade")

        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
            ),
        )

        cc_warp_size = 32
        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            cc_warp_size,
        )

        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"]
        }
        compile_kwargs = {
            "target": target,
            "options": options,
        }

        try:
            binary = triton.compile(*compile_args, **compile_kwargs)
        except Exception:
            log.debug(
                "Triton compilation failed: %s\n%s\nmetadata: %s",
                self.inductor_meta.get("kernel_name", "triton_"),
                self.fn.src,
                compile_meta,
            )
            raise
        return TritonCompileResultNpu(binary, cfg, compile_meta, self.inductor_meta)

    def _make_launchers(self):
        if len(self.launchers) == len(self.compile_results):
            return

        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, self.triton_meta["device"]):
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())
            launchers = []
            exc = None
            exc_stack = ""
            for result in self.compile_results:
                try:
                    launchers.append(result.make_launcher())
                except Exception as e:
                    import traceback
                    exc_stack = traceback.format_exc()
                    exc = e

        if len(launchers) == 0:
            raise RuntimeError(f"No valid triton configs. {type(exc).__name__}: {exc}\n"
                               f"Stack trace: {exc_stack}")
        self.launchers = launchers

    def save_gpu_kernel(self, input_stream, input_launcher):
        self.save_npu_kernel(input_stream, input_launcher)

    def save_npu_kernel(self, input_stream, input_launcher):
        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name

        if key is None:
            raise RuntimeError("assert key is not None, kernel_name can not be None")
        params = {
            "mangled_name": (
                input_launcher.bin.metadata.name
                if hasattr(input_launcher.bin.metadata, "name")
                else input_launcher.bin.metadata["name"]
            ),
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
    def bench(self, launcher, *args, with_profiler=False, **kwargs):
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

    def fallback_to_fx(self, *args, launcher, stream, **kwargs):
        """
        Try to fallback kernel to fx graph call according to kernel id.
        """
        def should_fallback():
            fallback_id = npu_config.force_fallback_kernel_id
            if fallback_id != "all" and not isinstance(fallback_id, list):
                raise RuntimeError("torch_npu._inductor.config.aot_inductor.force_fallback_kernel_id "
                                   "should be set to 'all' or List, e.g, [1, 2, 10]." + pta_error(ErrCode.VALUE))

            if isinstance(fallback_id, list):
                kernel_name = self.get_fn_name()
                try:
                    kernel_id = int(kernel_name.split("_")[-1])
                except ValueError:
                    kernel_id = -1
                if kernel_id not in fallback_id:
                    return False
            return True        

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
        

    def check_accuracy(self, *args, launcher, stream, **kwargs):
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

    def debug_kernel_in_run(self, *args, launcher, stream, **kwargs):
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

        result = super().run(*args, stream=stream, **kwargs)

        torch.npu.synchronize()
        torch.save(dump_args, f"{dump_path}/{idx}_{fn_name}_after.pt")
        return result


    def run(
        self, *args, stream, benchmark_run=False, **kwargs
    ):  # type:ignore[override]
        if self.triton_interpret:
            args, grid = self._interpret_args_grid(args, self.configs[0])
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
                self.autotune_to_one_config(*args, **kwargs)

        if not getattr(
                self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            self.launchers = [
                self.coordinate_descent_tuning(
                    self.launchers[0], *args, **kwargs
                )
            ]

        (launcher, ) = self.launchers
        if launcher.store_cubin and (not benchmark_run or not self.cuda_kernel_saved):
            self.save_gpu_kernel(stream, launcher)

        if self.dump_launch_params:
            _dump_launch_params(args, kwargs, launcher, self.fn.__name__)

        if npu_config.check_accuracy:
            if self.check_accuracy(*args, launcher=launcher, stream=stream, **kwargs):
                return

        elif npu_config.dump_fx_graph:
            fx_graph_call, kernel_name, dump_path, _ = self.get_fx_graph_call()
            if not fx_graph_call:
                log.warning(f"data dump for kernel {kernel_name} failed!")
            else:
                self.data_dump(*args, dump_path=dump_path)

        elif npu_config.force_fallback_kernel_id:
            fallback_result = self.fallback_to_fx(*args, launcher=launcher, stream=stream, **kwargs)
            if fallback_result is not None:
                log.debug(f"fallback kernel {self.get_fn_name()} to fx graph call.")
                return
            else:
                log.warning(f"kernel {self.get_fn_name()} could not fallback to fx.")
        elif npu_config.aot_inductor.debug_kernel_in_run:
            return self.debug_kernel_in_run(*args, launcher=launcher, stream=stream, **kwargs)

        # it is faster than entering and exiting a context manager, even if the context
        # manager is a nullcontext.
        if autograd_profiler._is_profiler_enabled:
            with torch._C._profiler._RecordFunctionFast(
                    self.inductor_meta.get("kernel_name", "triton kernel"),
                    args,
                    {
                        "kernel_file": (self.filename or ""),
                        "kernel_hash": self.kernel_hash,
                        "kernel_backend": "triton",
                        "stream": stream,
                    },
            ):
                return launcher(
                    *args,
                    **kwargs,
                    stream=stream,
                )
        else:
            return launcher(
                *args,
                **kwargs,
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
