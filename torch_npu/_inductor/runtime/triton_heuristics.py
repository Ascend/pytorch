# This file is based on triton_heuristics with heuristics designed for NPU
import copy
import functools
from functools import lru_cache
import importlib
import logging
import dataclasses
import math
import os
import re
import sys
import time
import shutil
import hashlib
import csv
import uuid
import threading
from itertools import count
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING, Union, List
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch._logging import warning_once
import triton
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config
from torch._inductor.compile_fx import clone_preserve_strides
from torch._inductor.runtime.autotune_cache import AutotuneCache
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.runtime.runtime_utils import (
    create_bandwidth_info_str,
    get_num_bytes,
    next_power_of_2,
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
    config_to_dict,
    config_from_dict,
    FixedGrid,
    PrecomputedGrid,
    SequentialComboKernelGrid,
    PrecomputedGrid,
    Grid1D,
    Grid2D,
    Grid2DWithYZOverflow,
    Grid3D
)
from .symbolic_grouping import (
    UnsupportedGroupedPlan,
    bucketize,
    build_group_representatives,
    find_primary_feature_index,
    is_open_bucket_group,
)
# used for wrapper codegen, do not remove this
from torch._inductor.runtime.triton_heuristics import ( # noqa: F401
    fixed_config,
    user_autotune,
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

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from torch_npu._inductor.npu_compare import check_accuracy_triton

from ..codegen.tile_generator import TileGenerator
from ..codegen.triton_utils import NPUKernelType
from ..config import log, autotune_continue_on_failure
from .. import config as npu_config
from ..profiler import simple_trace_handler, mspti_batch_benchmark

kernel_idx = count()


class CompileThreadPool:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=npu_config.max_precompiled_thread_num)
        self.warmup()

    def warmup(self):
        event = threading.Event()

        def worker():
            event.wait()

        tasks = []
        for _ in range(npu_config.max_precompiled_thread_num):
            tasks.append(self.submit(worker))
        event.set()
        for future in tasks:
            future.result()

    def submit(self, fn, *args, **kwargs):
        return self.pool.submit(fn, *args, **kwargs)


compile_thread_pool = CompileThreadPool()


@contextmanager
def create_profiler(torch_path, wait=0, warmup=1, active=1, repeat=1, skip_first=1):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0, )
    profile_path = torch_path
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
            on_trace_ready=simple_trace_handler(profile_path),
            experimental_config=experimental_config) as prof:
        yield prof


def delete_file_base(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)


def read_device_time(torch_path, triton_only=True, return_list=True):
    for root, _, files in os.walk(torch_path):
        for file in files:
            if file != 'kernel_details.csv':
                continue
            target_file = os.path.join(root, file)
            with open(target_file, newline='') as csvfile:
                durations = []
                reader = csv.DictReader(csvfile)
                for row_read in reader:
                    durations.append(float(row_read['Duration(us)']))
            if return_list:
                return durations
            ret = sum(durations) / len(durations)
            return ret
    delete_file_base(torch_path)
    raise RuntimeError(f"Could not find kernel_details.csv from dir {torch_path}")


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(torch, return_mode)(times).item()


def do_bench_using_profiling_npu(fn, warmup=2, rep=10, grad_to_none=None, quantiles=None, return_mode="mean"):
    if return_mode not in ["min", "max", "mean", "median", "all"]:
        raise RuntimeError("return_mode must be one of 'min', 'max', 'mean', 'median', 'all'")

    stream = torch.npu.current_stream()
    stream.synchronize()

    # Warm-up
    for _ in range(warmup):
        fn()
    stream.synchronize()

    random_uuid = uuid.uuid4().hex
    md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()
    torch_path = os.path.join(os.getcwd(), "profile_result", f"triton_{md5_hash}")
    with create_profiler(torch_path, active=8) as prof:
        stream.synchronize()
        for _ in range(rep + 10):
            fn()
            prof.step()
        stream.synchronize()
    times = read_device_time(torch_path, triton_only=False, return_list=True)
    delete_file_base(torch_path)
    return _summarize_statistics(torch.tensor(times), quantiles, return_mode)


@dataclasses.dataclass
class GridNpu(GridExpr):
    numels: List[str] = None
    mode: Literal["python", "cpp"] = "python"

    def generate(self, meta: dict[str, int]) -> None:
        numel_args = []
        split_axis = meta.get("split_axis", None)
        split_blocks = meta.get("split_blocks", None)
        if split_axis is None or split_blocks is None:
            raise RuntimeError(
                f"Could not get split_axis or split_blocks from meta {meta}."
            )

        def grid_fn(i):
            if i >= len(split_axis):
                return "1"
            axis = split_axis[i]
            block = split_blocks[i]
            if block is None or block == 1:
                return self.numels[axis]
            if self.mode == "python":
                return f"({self.numels[axis]} + {block} - 1) // {block}"
            else:
                return f"(({self.numels[axis]} + ({block} - 1)) / ({block}))"

        self.x_grid = grid_fn(0)
        self.y_grid = grid_fn(1)
        self.z_grid = grid_fn(2)


def is_namedtuple_isinstance(obj):
    return (
        isinstance(obj, tuple) and
        hasattr(obj, '_fields') and
        hasattr(obj, '_asdict') and
        callable(getattr(obj, '_asdict'))
    )


class GridExprNpu(GridExpr):
    @staticmethod
    def from_meta_and_set_numel(
        inductor_meta: dict[str, Any],
        cfg: Union[Config, dict[str, int]],
        numels: List[str],
        mode: Literal["python", "cpp"] = "python",
    ) -> GridExpr:
        grid_type = inductor_meta["grid_type"]

        grid_cls = globals().get(grid_type)
        if not issubclass(grid_cls, GridNpu):
            grid = grid_cls(inductor_meta=inductor_meta, mode=mode)
            if isinstance(cfg, Config):
                cfg = config_to_dict(cfg)
            grid.generate(cfg)
            return grid

        grid = grid_cls(inductor_meta=inductor_meta, mode=mode, numels=numels)
        runtime_block_names = tuple(inductor_meta.get("runtime_block_arg_names", ()))
        if runtime_block_names:
            axis_names = tuple(inductor_meta.get("axis_names", ()))
            runtime_axes = []
            for block_name in runtime_block_names:
                axis_name = block_name.removesuffix("BLOCK").lower()
                if axis_name not in axis_names:
                    raise RuntimeError(
                        f"runtime block grid path could not resolve runtime block axis {axis_name}"
                    )
                runtime_axes.append(axis_names.index(axis_name))

            def runtime_grid_fn(i):
                if i >= len(runtime_axes):
                    return "1"
                axis = runtime_axes[i]
                block = runtime_block_names[i]
                if block is None or block == 1:
                    return numels[axis]
                if mode == "python":
                    return f"({numels[axis]} + {block} - 1) // {block}"
                return f"(({numels[axis]} + ({block} - 1)) / ({block}))"

            grid.x_grid = runtime_grid_fn(0)
            grid.y_grid = runtime_grid_fn(1)
            grid.z_grid = runtime_grid_fn(2)
            return grid

        if isinstance(cfg, Config):
            cfg = config_to_dict(cfg)
        grid.generate(cfg)
        return grid

    @staticmethod
    def from_grouped_meta_and_numel(
        inductor_meta: dict[str, Any],
        cfg: Union[Config, dict[str, int]],
        numels: List[str],
        runtime_block_names: tuple[str, ...],
        mode: Literal["python", "cpp"] = "python",
    ) -> GridExpr:
        grid_type = inductor_meta["grid_type"]
        grid_cls = globals().get(grid_type)
        if not issubclass(grid_cls, GridNpu):
            raise RuntimeError(
                f"grouped grid path expects GridNpu-compatible grid, got {grid_type}"
            )
        grid = grid_cls(inductor_meta=inductor_meta, mode=mode, numels=numels)
        axis_names = tuple(inductor_meta.get("axis_names", ()))
        runtime_axes = []
        for block_name in runtime_block_names:
            axis_name = block_name.removesuffix("BLOCK").lower()
            if axis_name not in axis_names:
                raise RuntimeError(
                    f"grouped grid path could not resolve runtime block axis {axis_name}"
                )
            runtime_axes.append(axis_names.index(axis_name))

        def grouped_grid_fn(i):
            if i >= len(runtime_axes):
                return "1"
            axis = runtime_axes[i]
            block = runtime_block_names[i]
            if block is None or block == 1:
                return numels[axis]
            if mode == "python":
                return f"({numels[axis]} + {block} - 1) // {block}"
            return f"(({numels[axis]} + ({block} - 1)) / ({block}))"

        grid.x_grid = grouped_grid_fn(0)
        grid.y_grid = grouped_grid_fn(1)
        grid.z_grid = grouped_grid_fn(2)
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
                    arg
                    for arg in def_args
                    if arg not in ("num_warps", "num_stages")
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
            runtime_block_names = tuple(
                self.inductor_meta.get("runtime_block_arg_names", ())
            )
            def_args = filter_launcher_def_args(
                fn.arg_names,
                cfg_dict,
                none_args,
                runtime_block_names,
            )

        if self.inductor_meta.get("group_enabled", False):
            runtime_block_names = tuple(
                self.inductor_meta.get("runtime_block_append_order", ())
            )
            missing_runtime_block_names = [
                name
                for name in runtime_block_names
                if name not in def_args or name not in call_args
            ]
            if missing_runtime_block_names:
                raise RuntimeError(
                    "Grouped launcher is missing runtime block args: "
                    f"{missing_runtime_block_names}"
                )

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )
        if is_namedtuple_isinstance(binary.packed_metadata):
            binary.packed_metadata = binary.packed_metadata._asdict()
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
        linear_mode = self.inductor_meta.get('inductor_ascend_linear_mode', 'no_linear')
        runtime_block_names = tuple(
            self.inductor_meta.get("runtime_block_arg_names", ())
        )
        grid = None
        if self.inductor_meta.get("group_enabled", False):
            grid = GridExprNpu.from_grouped_meta_and_numel(
                self.inductor_meta,
                cfg,
                numels,
                runtime_block_names=runtime_block_names,
            )
        elif linear_mode == 'no_linear' and not runtime_block_names:
            grid = GridExpr.from_meta(self.inductor_meta, cfg)
        else:
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
        launcher.runnable = True
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
        self._costmodel_runtime_args = ()
        self._costmodel_runtime_kwargs = {}
        self._costmodel_fallback_configs = None
        self.runtime_block_arg_names = tuple(
            (inductor_meta or {}).get("runtime_block_arg_names", ())
        )
        # candidate_plan keeps the full runtime candidate space, while
        # variant_launcher_map points to the deduplicated compiled variants.
        self.candidate_plan = None
        self.variant_launcher_map = {}
        self.compiled_candidate_entries = ()
        self.best_candidate_config = None
        self.best_runtime_blocks = ()
        self.best_launcher = None

    def _set_best_candidate(
        self,
        candidate,
        launcher,
        runtime_blocks=None,
    ):
        if runtime_blocks is None:
            runtime_blocks = tuple(
                value for _, value in candidate.get("runtime_blocks", ())
            )
        else:
            runtime_blocks = tuple(runtime_blocks)

        if "full_config" in candidate:
            self.best_candidate_config = config_from_dict(candidate["full_config"])
        self.best_runtime_blocks = runtime_blocks
        self.best_launcher = launcher

    def _build_runtime_launch_args(self, args, runtime_blocks: tuple[int, ...]):
        return (*args, *runtime_blocks)

    def _first_compiled_candidate_entry(self):
        if not self.compiled_candidate_entries:
            raise RuntimeError(
                f"No compiled runtime block candidates are available for [{self.get_fn_name()}]."
            )
        return self.compiled_candidate_entries[0]

    def _precompile_variant_configs(self, configs=None):
        configs = self.configs if configs is None else configs
        plan = self.candidate_plan if configs is self.configs else None
        if plan is None:
            plan = build_candidate_plan(configs, self.runtime_block_arg_names)
        candidate_count = len(plan.get("candidate_entries", ()))
        variant_count = len(plan["variant_order"])
        if candidate_count > variant_count:
            log.info(
                "runtime block compile dedup: kernel=%s full_config_count=%s compile_config_count=%s",
                self.get_fn_name(),
                candidate_count,
                variant_count,
            )
        # Precompile only compile-distinct variants; candidate-level BLOCK choices
        # are still selected later during costmodel/autotune.
        variant_configs = []
        for variant_id in plan["variant_order"]:
            variant_cfg = config_from_dict(plan["variants"][variant_id]["config"])
            setattr(variant_cfg, "_runtime_variant_id", variant_id)
            variant_configs.append(variant_cfg)
        return variant_configs

    def _refresh_variant_launchers(self):
        self.variant_launcher_map = {}
        self.compiled_candidate_entries = ()
        if self.candidate_plan is None:
            self.candidate_plan = build_candidate_plan(
                self.configs, self.runtime_block_arg_names
            )
        launchers_by_variant_id = {}
        launchers_by_config = {
            repr(config_to_dict(launcher.config)): launcher for launcher in self.launchers
        }
        for launcher in self.launchers:
            variant_id = getattr(launcher.config, "_runtime_variant_id", None)
            if variant_id is not None and variant_id not in launchers_by_variant_id:
                launchers_by_variant_id[variant_id] = launcher
        for variant_id in self.candidate_plan["variant_order"]:
            launcher = launchers_by_variant_id.get(variant_id)
            if launcher is None:
                variant_config = self.candidate_plan["variants"][variant_id]["config"]
                launcher = launchers_by_config.get(repr(variant_config))
            if launcher is not None:
                self.variant_launcher_map[variant_id] = launcher
                continue
            log.debug(
                "No compiled launcher found for variant %s of kernel %s",
                variant_id,
                self.get_fn_name(),
            )
        self.compiled_candidate_entries = tuple(
            {
                "candidate": candidate,
                "launcher": self.variant_launcher_map[candidate["variant_id"]],
            }
            for candidate in self.candidate_plan["candidate_entries"]
            if candidate["variant_id"] in self.variant_launcher_map
        )

    def precompile(
        self,
        warm_cache_only=False,
        reload_kernel: Optional[Callable[[], CachingAutotuner]] = None,
        static_triton_bundle_key: Optional[str] = None,
    ):
        runtime_args, runtime_kwargs = self._resolve_costmodel_runtime_inputs()
        self._apply_costmodel_to_configs(*runtime_args, **runtime_kwargs)
        if self.candidate_plan is None:
            self.candidate_plan = build_candidate_plan(
                self.configs, self.runtime_block_arg_names
            )
        if warm_cache_only:
            self.kernel_name = self.get_fn_name()
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
        self._refresh_variant_launchers()

    def _make_ttir_module_from_cfg(self, cfg):
        """Compile one config to TTIR module only (no backend lowering/launch)."""
        compile_meta = copy.deepcopy(self.triton_meta)
        cfg_kwargs = cfg.kwargs
        constexpr_arg_names = {self.fn.arg_names[i] for i in self.fn.constexprs}
        for k, v in cfg_kwargs.items():
            if k not in constexpr_arg_names:
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

        compile_meta['compile_mode'] = cfg_kwargs.get('compile_mode')

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if not ASTSource:
            raise RuntimeError(
                "Installed triton version too old, please upgrade")

        src_attrs = compile_meta["configs"][0] if compile_meta.get(
            "configs", None) else None
        src = ASTSource(self.fn, compile_meta["signature"],
                        compile_meta["constants"], src_attrs)
        cc_warp_size = 32
        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            cc_warp_size,
        )
        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            "multibuffer": cfg_kwargs.get('multibuffer', False),
            "compile_mode": compile_meta['compile_mode'],
            "enable_vf_fusion": cfg_kwargs.get('enable_vf_fusion', False),
        }
        # pure simt stack overflow check
        if compile_meta['compile_mode'] == NPUKernelType.SIMT_ONLY.compile_mode():
            options['simt_stack_limit'] = npu_config.simt_default_warp_stacksize

        from triton.compiler.code_generator import ast_to_ttir
        from triton.compiler.compiler import make_backend
        from triton._C.libtriton import ir
        from triton._C.libtriton.ascend import ir as ascend_ir

        backend = make_backend(target)
        extra_options = src.parse_options()
        options = backend.parse_options(dict(options or dict(), **extra_options))
        context = ir.context()
        ir.load_dialects(context)
        ascend_ir.load_dialects(context)
        return ast_to_ttir(self.fn, src, context, options, {}, {})

    def _build_costmodel_runtime_args_from_size_hints(self):
        """Best-effort fallback runtime args when precompile is called without run()."""
        size_hints = getattr(self, "size_hints", None)
        if size_hints is None:
            return ()

        hint_map = {}
        if isinstance(size_hints, dict):
            hint_map.update(size_hints)
        elif isinstance(size_hints, (list, tuple)):
            axis_names = []
            if isinstance(getattr(self, "inductor_meta", None), dict):
                axis_names = list(self.inductor_meta.get("axis_names", []) or [])
            if axis_names and len(axis_names) == len(size_hints):
                for axis, val in zip(axis_names, size_hints):
                    hint_map[axis] = val
                    hint_map[f"{axis}_numel"] = val

        signature_names = list(self.triton_meta.get("signature", {}).keys())
        if not signature_names:
            return ()

        runtime_args = []
        for name in signature_names:
            key = str(name)
            if key.endswith("_numel"):
                val = hint_map.get(key)
                if val is None:
                    val = hint_map.get(key[:-6])
                int_val = self._try_parse_int_like(val)
                runtime_args.append(int_val if int_val is not None else 1)
            else:
                runtime_args.append(object())
        return tuple(runtime_args)

    def _resolve_costmodel_runtime_inputs(self):
        runtime_args = getattr(self, "_costmodel_runtime_args", ())
        runtime_kwargs = getattr(self, "_costmodel_runtime_kwargs", {})
        if runtime_args or runtime_kwargs:
            return runtime_args, runtime_kwargs

        fallback_args = self._build_costmodel_runtime_args_from_size_hints()
        if fallback_args:
            return fallback_args, {}
        return (), {}

    def _try_parse_int_like(self, value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except Exception:
            return None

    def _build_ttir_arg_value_map(self, ttir_text, runtime_args, runtime_kwargs=None, cfg=None):
        """Map TTIR arg id -> runtime value.

        Primary mapping uses frontend signature names (positional + kwargs) so callers
        can pass named args without relying on positional order only.
        """
        if runtime_kwargs is None:
            runtime_kwargs = {}
        cfg_kwargs = getattr(cfg, "kwargs", {}) if cfg is not None else {}

        m = re.search(r"tt\.func\s+public\s+@\w+\((.*?)\)\s+attributes", ttir_text, re.S)
        if m is None:
            return {}

        ttir_arg_ids = [int(x) for x in re.findall(r"%arg(\d+)\s*:", m.group(1))]
        if not ttir_arg_ids:
            return {}

        signature_names = list(self.triton_meta.get("signature", {}).keys())
        name_to_value = {}
        for idx, name in enumerate(signature_names):
            if idx < len(runtime_args):
                name_to_value[name] = runtime_args[idx]
        for name, value in runtime_kwargs.items():
            if name in name_to_value or name in signature_names:
                name_to_value[name] = value
        for name, value in cfg_kwargs.items():
            if name not in signature_names:
                continue
            current_value = name_to_value.get(name)
            if name not in name_to_value or self._try_parse_int_like(current_value) is None:
                name_to_value[name] = value

        arg_value_map = {}
        for pos, arg_id in enumerate(ttir_arg_ids):
            if pos < len(signature_names):
                name = signature_names[pos]
                if name in name_to_value:
                    arg_value_map[arg_id] = name_to_value[name]
                    continue
            if pos < len(runtime_args):
                arg_value_map[arg_id] = runtime_args[pos]
        return arg_value_map

    def _build_costmodel_arg_bindings(self, ttir_text, runtime_args, runtime_kwargs=None, cfg=None):
        """Build costmodel arg-bindings string from TTIR arg ids and runtime values."""
        try:
            arg_value_map = self._build_ttir_arg_value_map(ttir_text, runtime_args, runtime_kwargs, cfg)
            if not arg_value_map:
                return ""

            bindings = []
            for arg_id in sorted(arg_value_map.keys()):
                int_value = self._try_parse_int_like(arg_value_map[arg_id])
                if int_value is None:
                    continue
                bindings.append(f"arg{arg_id}={int_value}")

            # Use a stable default program-id binding for static estimation.
            bindings.append("pid_x=0")

            # Bind tt.get_num_programs when present in TTIR.
            if "tt.get_num_programs x" in ttir_text:
                num_programs_x = None
                if runtime_kwargs:
                    if "num_programs_x" in runtime_kwargs:
                        num_programs_x = self._try_parse_int_like(runtime_kwargs.get("num_programs_x"))
                    elif "grid" in runtime_kwargs:
                        grid = runtime_kwargs.get("grid")
                        if isinstance(grid, (tuple, list)) and len(grid) > 0:
                            num_programs_x = self._try_parse_int_like(grid[0])
                if num_programs_x is None:
                    num_programs_x = 1
                bindings.append(f"num_programs_x={num_programs_x}")

            return ",".join(bindings)
        except Exception:
            return ""

    def _build_costmodel_items(self, runtime_args, runtime_kwargs=None):
        """Build config->TTIR payloads for costmodel evaluation."""
        if runtime_kwargs is None:
            runtime_kwargs = {}
        items = []
        plan = build_candidate_plan(self.configs, self.runtime_block_arg_names)
        ttir_text_by_variant = {}

        candidate_entries = tuple(plan.get("candidate_entries", ()))
        for cfg, candidate in zip(self.configs, candidate_entries):
            variant_id = candidate["variant_id"]
            ttir_text = ttir_text_by_variant.get(variant_id)
            try:
                if ttir_text is None:
                    # TTIR is shared by compile-equivalent variants; runtime BLOCK
                    # values are rebound per candidate through arg_bindings.
                    variant_cfg = config_from_dict(
                        plan["variants"][variant_id]["config"]
                    )
                    ttir_module = self._make_ttir_module_from_cfg(variant_cfg)
                    ttir_text = str(ttir_module)
                    ttir_text_by_variant[variant_id] = ttir_text
                candidate_runtime_kwargs = dict(runtime_kwargs)
                candidate_runtime_kwargs.update(dict(candidate["runtime_blocks"]))
                arg_bindings = self._build_costmodel_arg_bindings(
                    ttir_text, runtime_args, candidate_runtime_kwargs
                )
                items.append(
                    {
                        "config": cfg,
                        "ttir": ttir_text,
                        "arg_bindings": arg_bindings,
                    }
                )
            except Exception:
                items.append({"config": cfg, "ttir": "", "arg_bindings": ""})
        return items

    def _select_ttir_test_config(self):
        smallest_config = None
        min_sub_product = float("inf")
        for cfg in self.configs:
            kwargs = getattr(cfg, "kwargs", None) or {}
            current_sub_product = 1
            has_sub_tiling = False
            for tiling_name, tiling in kwargs.items():
                if not isinstance(tiling_name, str) or not tiling_name.endswith("SUB"):
                    continue
                try:
                    tiling_value = int(tiling)
                except (TypeError, ValueError):
                    continue
                current_sub_product *= tiling_value
                has_sub_tiling = True
            if has_sub_tiling and current_sub_product < min_sub_product:
                min_sub_product = current_sub_product
                smallest_config = cfg
        return smallest_config if smallest_config is not None else self.configs[0]

    def _triton_make_ttir(self):
        if not self.configs:
            raise NoTritonConfigsError("No triton configs are available")

        def make_ttir_from_cfg(cfg):
            """Ahead of time compile a given autotuner config."""
            try:
                ttir_module = self._make_ttir_module_from_cfg(cfg)
                log.debug(
                    "Triton precompile success: %s\n%s\nmodule: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    str(ttir_module),
                )
            except Exception as e:
                log.debug(
                    "Triton precompile failed: %s\n%s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                )
                raise e

            return ttir_module

        compile_results = []
        exc = None
        exc_stack = ""
        test_config = self._select_ttir_test_config()
        try:
            compile_results.append(make_ttir_from_cfg(test_config))
        except Exception as e:
            import traceback
            exc_stack = traceback.format_exc()
            exc = e
        if len(compile_results) == 0:
            raise NoTritonConfigsError(
                f"Failed to compile config {test_config} for make ttir {self.fn.__name__}: {exc} \nStack trace:{exc_stack}"
            )
        self.compile_results = compile_results

    def _apply_costmodel_to_configs(self, *args, **kwargs):
        """Use triton-ascend costmodel path to prefilter configs before full compile."""
        self._costmodel_fallback_configs = None
        if self.heuristic_type == HeuristicType.USER_AUTOTUNE:
            return

        if not self.configs or len(self.configs) <= 1:
            return

        if not bool(npu_config.enable_costmodel_backend):
            return

        costmodel_ratio = float(getattr(npu_config, "costmodel_ratio", 0.25))
        if costmodel_ratio <= 0.0 or costmodel_ratio >= 1.0:
            return

        try:
            from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
            costmodel_items = self._build_costmodel_items(args, kwargs)
            costmodel_start_time = time.perf_counter()
            costmodel_map = costmodel_bench(costmodel_items)
            costmodel_elapsed = time.perf_counter() - costmodel_start_time
            log.debug("costmodel_bench elapsed time: %.6fs", costmodel_elapsed)
            log.debug("costmodel_bench result: %s", costmodel_map)
        except Exception as exc:
            log.warning("Skip costmodel prefilter because costmodel path is unavailable: %s", exc)
            return

        if not isinstance(costmodel_map, dict):
            return

        ranked_cfgs = [
            cfg for cfg, t in sorted(costmodel_map.items(), key=lambda kv: kv[1])
            if t != float("inf")
        ]
        target_count = max(1, math.ceil(len(self.configs) * costmodel_ratio))
        if len(ranked_cfgs) < target_count:
            log.warning("Config count filtered by costmodel is less than target count")
        if ranked_cfgs:
            selected_count = min(target_count, len(ranked_cfgs))
            self.configs = ranked_cfgs[:selected_count]
            self._costmodel_fallback_configs = ranked_cfgs[selected_count:] or None

    def _precompile_configs(self, configs):
        if not configs:
            raise NoTritonConfigsError("No triton configs are available")

        compile_results = []
        exc = None
        exc_stack = ""
        compile_start_time = time.perf_counter()
        for c in configs:
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
        log.info(f"kernel: {self.get_fn_name()} compile cost time: {time.perf_counter() - compile_start_time}s")
        return compile_results

    def _precompile_with_costmodel_fallback(self, compile_fn):
        primary_configs = self.configs
        primary_plan = self.candidate_plan
        if primary_plan is None:
            primary_plan = build_candidate_plan(
                primary_configs, self.runtime_block_arg_names
            )
        primary_compile_configs = self._precompile_variant_configs(primary_configs)
        try:
            compile_results = compile_fn(primary_compile_configs)
            self.compile_results = compile_results
            self.configs = primary_configs
            self.candidate_plan = primary_plan
            return
        except NoTritonConfigsError as primary_exc:
            fallback_configs = getattr(self, "_costmodel_fallback_configs", None)
            if not fallback_configs:
                raise
            log.warning(
                "No valid triton configs from costmodel-selected configs for kernel %s; "
                "retrying %d costmodel-filtered configs.",
                self.get_fn_name(),
                len(fallback_configs),
            )
            fallback_plan = build_candidate_plan(
                fallback_configs, self.runtime_block_arg_names
            )
            fallback_compile_configs = self._precompile_variant_configs(fallback_configs)
            try:
                compile_results = compile_fn(fallback_compile_configs)
                self.compile_results = compile_results
                self.configs = fallback_configs
                self.candidate_plan = fallback_plan
                return
            except NoTritonConfigsError as fallback_exc:
                raise NoTritonConfigsError(
                    f"No valid triton configs from costmodel-selected or fallback configs. "
                    f"Primary error: {primary_exc}. Fallback error: {fallback_exc}"
                ) from fallback_exc

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

        self._precompile_with_costmodel_fallback(self._precompile_configs)
        self._costmodel_fallback_configs = None

    def parse_triton_ascend_options(self, tiling_kwargs, options):
        from triton.backends.ascend.compiler import NPUOptions
        for k in NPUOptions.__dataclass_fields__.keys():
            if k not in tiling_kwargs:
                continue
            options[k] = tiling_kwargs[k]

        return options

    def _precompile_config(self, cfg: Config) -> TritonCompileResultNpu:
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        cfg_kwargs = cfg.kwargs
        constexpr_arg_names = {self.fn.arg_names[i] for i in self.fn.constexprs}
        for k, v in cfg_kwargs.items():
            if k not in constexpr_arg_names:
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

        compile_meta['compile_mode'] = cfg_kwargs.get('compile_mode')

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if not ASTSource:
            raise RuntimeError("Installed triton version too old, please upgrade")

        if compile_meta.get("configs", None):
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                    compile_meta["configs"][0],
                ),
            )
        else:
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
            "debug": compile_meta["debug"],
            "compile_mode": compile_meta['compile_mode'],
        }

        options = self.parse_triton_ascend_options(cfg_kwargs, options)
        # pure simt stack overflow check
        if compile_meta['compile_mode'] == NPUKernelType.SIMT_ONLY.compile_mode():
            options['simt_stack_limit'] = npu_config.simt_default_warp_stacksize
        if self.inductor_meta.get("inductor_ascend_linear_mode", "no_linear") == "no_linear":
            options['enable_auto_blockify'] = True

        compile_kwargs = {
            "target": target,
            "options": options,
        }
        start_time = 0
        if log.isEnabledFor(logging.DEBUG):
            start_time = time.perf_counter()
        try:
            binary = None
            binary = triton.compile(*compile_args, **compile_kwargs)
            required_ub_bits = binary.metadata.required_ub_bits
            cfg.real_ub_size = required_ub_bits

        except AttributeError as e:
            # 该错误表示编译无法通过（一些情况下也会被这个Exception捕获）
            log.debug(f"config: {cfg_kwargs} compile failed, cost time: {time.perf_counter() - start_time}s")
            if binary is None:
                raise Exception("Triton compilation failed") from e

            # mistake is metadata no required_ub_bits, but binary work.
            return TritonCompileResultNpu(binary, cfg, compile_meta, self.inductor_meta)

        except Exception:
            log.debug(
                "Triton compilation failed: %s\n%s\nmetadata: %s",
                self.inductor_meta.get("kernel_name", "triton_"),
                self.fn.src,
                compile_meta,
            )
            log.debug(f"config: {cfg_kwargs} compile failed, cost time: {time.perf_counter() - start_time}s")
            import traceback
            ts = traceback.format_exc()
            match = re.search(r"ub overflow.*?requires (\d+) bits", ts)
            if match:
                required_ub_bits = int(match.group(1))
                cfg.real_ub_size = required_ub_bits
            raise
        log.debug(f"config: {cfg_kwargs} compile success, cost time: {time.perf_counter() - start_time}s")
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

    def save_gpu_kernel(self, stream, launcher):
        self.save_npu_kernel(stream, launcher)

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
            "config": config_to_dict(input_launcher.config),
            "inductor_meta": self.inductor_meta,
            "triton_meta": self.triton_meta,
            "def_args": input_launcher.def_args,
            "call_args": input_launcher.call_args,
            "runtime_blocks": dict(
                zip(self.runtime_block_arg_names, self.best_runtime_blocks)
            ),
            "mix_mode": input_launcher.bin.metadata.mix_mode,
            "parallel_mode": input_launcher.bin.metadata.parallel_mode,
            "force_simt_only": input_launcher.bin.metadata.force_simt_only
        }
        enable_simt = ("simt" in params["parallel_mode"]) or params["force_simt_only"]
        if npu_config.is_ascend950 and enable_simt:
            params["shared_mem_dynamic_size"] = input_launcher.bin.metadata.shared_mem_dynamic_size

        from torch._inductor.codecache import CudaKernelParamCache

        bin_type = "npubin"
        binary = input_launcher.bin.asm[bin_type]  # npubin type = npubin
        CudaKernelParamCache.set(key, params, binary, bin_type='cubin')  # CudaKernelParam

        self.cuda_kernel_saved = True

    def _precompile_configs_parallel(self, configs):
        if not configs:
            raise NoTritonConfigsError("No triton configs are available")

        config_len = len(configs)
        compile_exc_results = [None for _ in range(config_len)]
        compile_exc_stack_results = ["" for _ in range(config_len)]

        def worker(i, kernel_config):
            try:
                return self._precompile_config(kernel_config)
            except Exception as e:
                import traceback
                compile_exc_stack_results[i] = traceback.format_exc()
                compile_exc_results[i] = e
                return None

        tasks = []
        for i, c in enumerate(configs):
            task_handler = compile_thread_pool.submit(worker, i, c)
            tasks.append(task_handler)

        from torch._dynamo.device_interface import DeviceGuard
        device_interface = self.get_device_interface()
        # load binary to the correct device
        compile_results = []
        with DeviceGuard(device_interface, self.triton_meta["device"]):
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())
            for future in as_completed(tasks):
                compiled_kernel = future.result()
                if compiled_kernel is None:
                    continue
                compile_results.append(compiled_kernel)

        # first try but return no valid configs
        # so we try tuning more options
        if len(compile_results) == 0:
            # set up new configs
            for i in range(len(configs)):
                # in future, adjust more options
                configs[i].kwargs["enable_vf_fusion"] = True
            # start compilation tasks
            tasks = []
            for i, c in enumerate(configs):
                task_handler = compile_thread_pool.submit(worker, i, c)
                tasks.append(task_handler)
            # collect compiled results
            with DeviceGuard(device_interface, self.triton_meta["device"]):
                # need to initialize context
                device_interface.synchronize(device_interface.current_device())
                for future in as_completed(tasks):
                    compiled_kernel = future.result()
                    if compiled_kernel is None:
                        continue
                    compile_results.append(compiled_kernel)

        if len(compile_results) == 0:
            raise NoTritonConfigsError(
                f"No valid triton configs for kernel {self.get_fn_name()}. {type(compile_exc_results[0]).__name__}: {compile_exc_results[0]} \nStack trace:{compile_exc_stack_results[0]}"
            )
        return compile_results

    def _precompile_worker_parallel(self):
        if self.compile_results:
            for result in self.compile_results:
                TritonBundler.put(
                    triton_hash_to_path_key(result.kernel.hash),
                    self.triton_meta.get("device", 0),
                )
            return

        if self.launchers:
            raise AssertionError("Before _precompile_worker, launchers must bt empty")

        self._precompile_with_costmodel_fallback(self._precompile_configs_parallel)
        self._costmodel_fallback_configs = None

    # bench method is called by torch, grid can not be modified
    def bench(self, launcher, *args, with_profiler=False, runtime_blocks=None, **kwargs):
        """Measure the performance of a given launcher"""

        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
                "spill_threshold", 16
        ):
            return float("inf")

        if runtime_blocks is None:
            runtime_blocks = (
                self.best_runtime_blocks if launcher is self.best_launcher else ()
            )
        launch_args = self._build_runtime_launch_args(args, tuple(runtime_blocks))
        return self._bench_with_launch_args(
            launcher,
            launch_args,
            args,
            **kwargs,
        )

    def _bench_with_launch_args(self, launcher, launch_args, reset_args, **kwargs):
        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        def kernel_call():
            cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
            self.reset_to_zero_args(*reset_args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                stream=stream,
            )

        if self.inductor_meta.get("profile_bandwidth_with_do_bench_using_profiling", False):
            return do_bench_using_profiling_npu(kernel_call, rep=1)

        return benchmarker.benchmark_gpu(kernel_call, rep=1)

    def _profile_batch_benchmark(self, kernel_funcs):
        def delete_file(base_path):
            if os.path.exists(base_path):
                shutil.rmtree(base_path)

        stream = torch.npu.current_stream()
        random_uuid = uuid.uuid4().hex
        md5_hash = hashlib.md5(random_uuid.encode()).hexdigest()

        kernel_count = len(kernel_funcs)
        autotune_path = os.path.join(os.getcwd(), "profile_result", f"triton_{md5_hash}")
        WAIT = 1
        WARMUP = 1
        ACTIVE = 10
        REPEAT = 1
        SKIP_FIRST = 1
        REDUNDANT_STEP = 3  # Add a few redundant steps to ensure profiler stability and sufficient triton kernel profiling data
        TOTAL_STEP = (WAIT + WARMUP + ACTIVE + SKIP_FIRST + REDUNDANT_STEP) * REPEAT
        with create_profiler(autotune_path, WAIT, WARMUP, ACTIVE, REPEAT, SKIP_FIRST) as prof:
            stream.synchronize()
            for _ in range(TOTAL_STEP):
                for fn in kernel_funcs:
                    fn()
                torch.npu.synchronize()
                prof.step()
            stream.synchronize()

        import pandas as pd
        for root, _, files in os.walk(autotune_path):
            for file in files:
                if file != 'kernel_details.csv':
                    continue
                target_file = os.path.join(root, file)
                df = pd.read_csv(target_file)
                triton_rows = df[df['Name'].str.startswith('triton', na=False)]
                if len(triton_rows) != kernel_count * ACTIVE:
                    raise RuntimeError(
                        f"Expected {kernel_count * ACTIVE} rows for triton kernels, "
                        f"but got {len(triton_rows)}. This may be due to profiling "
                        f"errors. Please check the profiling result at {target_file} "
                        "for more details."
                    )

                time_cost = [0] * kernel_count
                for kernel_index in range(kernel_count):
                    for active_index in range(ACTIVE):
                        row_index = kernel_index + kernel_count * active_index
                        time_cost[kernel_index] += triton_rows.iloc[row_index]['Duration(us)']
                delete_file(autotune_path)
                return [cost / ACTIVE for cost in time_cost]

        delete_file(autotune_path)
        return []

    def _benchmark_kernel_funcs_batch(self, kernel_funcs, benchmark_label):
        if not kernel_funcs or not npu_config.aggresive_autotune:
            return None

        try:
            batch_timings = tuple(
                mspti_batch_benchmark(kernel_funcs, filter_list=["triton"])
            )
            if len(batch_timings) != len(kernel_funcs):
                raise RuntimeError(
                    f"{benchmark_label} mspti batch timing count does not match kernel funcs"
                )
            return batch_timings
        except Exception as mspti_exc:
            log.warning(
                "%s mspti batch benchmark failed, switched to profiler batch: %s",
                benchmark_label,
                mspti_exc,
            )

        try:
            batch_timings = tuple(self._profile_batch_benchmark(kernel_funcs))
            if len(batch_timings) != len(kernel_funcs):
                raise RuntimeError(
                    f"{benchmark_label} profiler batch timing count does not match kernel funcs"
                )
            return batch_timings
        except Exception as profile_exc:
            log.warning(
                "%s profiler batch benchmark failed, switched to single bench: %s",
                benchmark_label,
                profile_exc,
            )
            return None

    def _benchmark_candidate_entries(self, *args, **kwargs):
        plan = self.candidate_plan
        if not self.variant_launcher_map:
            self._refresh_variant_launchers()
        log.info(
            f"{self.get_fn_name()} candidate entry count = {len(plan['candidate_entries'])}"
        )
        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())
        timings = {
            candidate["candidate_id"]: float("inf")
            for candidate in plan["candidate_entries"]
        }
        runnable_candidates = []
        prerun_timings = []
        candidate_map = {
            candidate["candidate_id"]: candidate
            for candidate in plan["candidate_entries"]
        }
        compiled_candidate_ids = {
            entry["candidate"]["candidate_id"]
            for entry in self.compiled_candidate_entries
        }
        for candidate in plan["candidate_entries"]:
            candidate_id = candidate["candidate_id"]
            if candidate_id in compiled_candidate_ids:
                continue
            log.debug(
                "No compiled launcher found for candidate %s (variant %s) of kernel %s",
                candidate_id,
                candidate["variant_id"],
                self.get_fn_name(),
            )

        # Benchmarking stays at candidate granularity even when several candidates
        # share the same compiled launcher through variant deduplication.
        for entry in self.compiled_candidate_entries:
            candidate = entry["candidate"]
            launcher = entry["launcher"]
            runtime_blocks = tuple(value for _, value in candidate["runtime_blocks"])
            candidate_id = candidate["candidate_id"]
            launch_args = self._build_runtime_launch_args(args, runtime_blocks)

            if not self.custom_kernel and launcher.n_spills > config.triton.spill_threshold:
                continue

            # Freeze loop variables so the batch benchmark keeps per-candidate args.
            def prerun(launcher=launcher, launch_args=launch_args):
                if launcher.config.pre_hook is not None:
                    launcher.config.pre_hook(
                        {
                            **dict(zip(self.arg_names, launch_args)),
                            **launcher.config.kwargs,
                        }
                    )
                cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
                self.reset_to_zero_args(*args, **kwargs)
                launcher(
                    *cloned_args,
                    **cloned_kwargs,
                    stream=stream,
                )

            try:
                prerun_ms = _measure_prerun_ms(prerun)
                prerun_timings.append((candidate_id, prerun_ms))
                runnable_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "launcher": launcher,
                        "runtime_blocks": runtime_blocks,
                        "prerun": prerun,
                    }
                )
                log.debug(
                    f"PreRun [{self.fn.__name__}], candidate: {candidate_id}\n"
                    f" variant [{launcher.config}] runtime_blocks [{runtime_blocks}] "
                    f"success, elapsed: {prerun_ms:.3f} ms"
                )
            except Exception as e:
                prerun_log_str = (
                    f"PreRun [{self.fn.__name__}], candidate: {candidate_id}\n"
                    f" variant [{launcher.config}] runtime_blocks [{runtime_blocks}] \n err: {e}"
                )
                if autotune_continue_on_failure:
                    log.warning(prerun_log_str)
                else:
                    raise RuntimeError(prerun_log_str)

        selected_candidate_ids = _select_prerun_top_candidates(prerun_timings)
        selected_runnable_candidates = [
            entry
            for entry in runnable_candidates
            if entry["candidate_id"] in selected_candidate_ids
        ]

        selected_total_ms = sum(
            prerun_ms
            for candidate_id, prerun_ms in prerun_timings
            if candidate_id in selected_candidate_ids
        )
        prerun_total_ms = sum(prerun_ms for _, prerun_ms in prerun_timings)
        log.info(
            "%s select %s candidates to %s, total candidate prerun time = %.3f ms, "
            "need rerun benchmark time = %.3f ms, saved benchmark time = %.3f ms",
            self.get_fn_name(),
            len(plan["candidate_entries"]),
            len(selected_runnable_candidates),
            prerun_total_ms,
            selected_total_ms,
            prerun_total_ms - selected_total_ms,
        )

        kernel_funcs = [entry["prerun"] for entry in selected_runnable_candidates]
        batch_timings = self._benchmark_kernel_funcs_batch(kernel_funcs, "candidate")
        if batch_timings is not None:
            for entry, timing in zip(selected_runnable_candidates, batch_timings):
                timings[entry["candidate_id"]] = timing
        else:
            for entry in selected_runnable_candidates:
                timings[entry["candidate_id"]] = self.bench(
                    entry["launcher"],
                    *args,
                    runtime_blocks=entry["runtime_blocks"],
                    **kwargs,
                )

        for candidate_id, timing in timings.items():
            candidate = candidate_map.get(candidate_id)
            if candidate is None:
                continue
            self.coordesc_tuner.cache_benchmark_result(
                config_from_dict(candidate["full_config"]), timing
            )

        if not timings or all(value == float("inf") for value in timings.values()):
            raise RuntimeError(
                f"All runtime block candidates for [{self.fn.__name__}] are not runnable."
            )
        if log.isEnabledFor(logging.DEBUG):
            sorted_timings = sorted(timings.items(), key=lambda item: item[1])
            for candidate_id, timing in sorted_timings:
                candidate = candidate_map[candidate_id]
                log.debug(
                    "[%s] [%s] runtime_blocks=%s benchmark time: [%f] us",
                    self.fn.__name__,
                    candidate["full_config"],
                    candidate["runtime_blocks"],
                    timing,
                )
        return timings

    def _should_skip_autotune_for_determinism(self):
        """
        When deterministic algorithms are enabled, skip benchmarking for
        reduction kernels and use the first successfully compiled config.
        All configs are still compiled so that compilation failures on NPU
        do not crash the program.
        Pointwise kernels are unaffected because their tiling does not change
        floating-point numerics.
        """
        return (
            self.inductor_meta.get("are_deterministic_algorithms_enabled")
            and self.heuristic_type in (
                HeuristicType.REDUCTION,
                HeuristicType.PERSISTENT_REDUCTION,
            )
        )

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        start_time = time.time_ns()
        if self.candidate_plan is None:
            self.candidate_plan = build_candidate_plan(
                self.configs, self.runtime_block_arg_names
            )
        timings = self._benchmark_candidate_entries(*args, **kwargs)
        benchmark_time_taken_ns = time.time_ns() - start_time
        candidate_map = {
            candidate["candidate_id"]: candidate
            for candidate in self.candidate_plan["candidate_entries"]
        }
        best_candidate_id, _ = builtins.min(
            timings.items(), key=lambda item: item[1]
        )
        best_candidate = candidate_map[best_candidate_id]
        if not self.variant_launcher_map:
            self._refresh_variant_launchers()
        best_launcher = self.variant_launcher_map[best_candidate["variant_id"]]
        self._set_best_candidate(best_candidate, best_launcher)
        self.launchers = [best_launcher]
        self.autotune_time_taken_ns = (
                self.precompile_time_taken_ns + benchmark_time_taken_ns
        )
        if self.save_cache_hook:
            best_config = (
                self.best_candidate_config
                if self.best_candidate_config is not None
                else self.launchers[0].config
            )
            self.save_cache_hook(best_config, self.autotune_time_taken_ns)

    @lru_cache(None)
    def get_fx_graph_dump_path(self):
        traced_graph_hash = self.inductor_meta.get("traced_graph_hash")
        if not traced_graph_hash:
            return None
        dump_dir = self.inductor_meta.get("traced_graph_dir", "")
        dump_path = os.path.join(dump_dir, traced_graph_hash)
        if dump_dir == "" or not os.path.exists(dump_path):
            return None
        return dump_path

    def data_dump(self, *args, dump_path=None):
        dump_path = self.get_fx_graph_dump_path() if dump_path is None else dump_path
        if dump_path is None:
            log.warning(f"data dump for kernel {self.get_fn_name()} failed, no valid dump_path is supplied.")
            return False
        data_dump_path = os.path.join(dump_path, 'data.pth')
        torch.save(args, data_dump_path)
        return True

    def get_fn_name(self):
        if self.fn_name is not None:
            return self.fn_name
        try:
            self.fn_name = self.fn.fn.__name__
        except AttributeError:
            self.fn_name = "unknown"
            if hasattr(self, 'kernel_name'):
                self.fn_name = self.kernel_name
        return self.fn_name

    @functools.lru_cache(None)
    def is_run_debug(self):
        return npu_config.dump_fx_graph or npu_config.check_accuracy

    def maybe_run_debug(self, *args, grid_, stream, launcher, **kwargs):
        kernel_name = self.get_fn_name()
        log.info(f"Try to run debug mode for kernel {kernel_name}.")
        if npu_config.dump_fx_graph:
            if torch_npu.npu.is_current_stream_capturing():
                raise RuntimeError(
                    "INDUCTOR_ASCEND_CHECK_ACCURACY / INDUCTOR_ASCEND_DUMP_FX_GRAPH "
                    "is not compatible with aclgraph.\n"
                    "Please disable aclgraph before enabling INDUCTOR_ASCEND_CHECK_ACCURACY "
                    "/ INDUCTOR_ASCEND_DUMP_FX_GRAPH, or unset these environment variables."
                )
            _ = self.data_dump(*args)

        if npu_config.check_accuracy:
            if check_accuracy_triton(
                *args,
                launcher=launcher,
                grid=grid_,
                stream=stream,
                inductor_meta=self.inductor_meta,
                **kwargs
            ):
                return "check_accuracy"

        log.info(f"No debug mode is activated for kernel {kernel_name}.")
        return None

    def run(
        self, *args, stream, benchmark_run=False, **kwargs
    ):  # type:ignore[override]

        if self.triton_interpret:
            cfg = self.best_candidate_config or self.configs[0]
            runtime_blocks = self.best_runtime_blocks
            args, grid = self._interpret_args_grid(args, cfg, runtime_blocks)
            launch_args = self._build_runtime_launch_args(args, runtime_blocks)
            copied_kwargs = copy.copy(cfg.kwargs)
            copied_kwargs.pop('split_axis', None)
            copied_kwargs.pop('split_blocks', None)
            for name in self.runtime_block_arg_names:
                copied_kwargs.pop(name, None)

            return self.fn[grid](
                *launch_args,
                **kwargs,
                **copied_kwargs,
            )

        if len(self.launchers) == 1 and hasattr(self.launchers[0], "fallback"):
            return self.launchers[0](
                *args,
                **kwargs,
            )

        self.autotuner(*args, stream=stream, benchmark_run=benchmark_run, **kwargs)
        launcher = self.best_launcher if self.best_launcher is not None else self.launchers[0]
        runtime_blocks = self.best_runtime_blocks

        if not getattr(
                launcher.config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            launcher = self.coordinate_descent_tuning(
                launcher, *args, **kwargs
            )
            self.launchers = [launcher]
            if self.best_launcher is not None:
                self.best_launcher = launcher

        if launcher.store_cubin and (not benchmark_run or not self.cuda_kernel_saved):
            self.save_gpu_kernel(stream, launcher)

        if self.dump_launch_params:
            _dump_launch_params(args, kwargs, launcher, self.fn.__name__)

        launch_args = self._build_runtime_launch_args(args, runtime_blocks)
        if self.is_run_debug() and not self.heuristic_type == HeuristicType.USER_AUTOTUNE:
            debug_cfg = self.best_candidate_config or launcher.config
            _, grid = self._interpret_args_grid(args, debug_cfg, runtime_blocks)
            debug_mode = self.maybe_run_debug(*launch_args, grid_=grid, stream=stream, launcher=launcher, **kwargs)
            if debug_mode:
                log.info(f"Kernel {self.get_fn_name()} goes into {debug_mode} and return.")
                return

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
                    *launch_args,
                    **kwargs,
                    stream=stream,
                )
        else:
            return launcher(
                *launch_args,
                **kwargs,
                stream=stream,
            )

    def autotuner(self, *args, stream, benchmark_run=False, **kwargs):
        if self.best_launcher is not None:
            return
        if self.candidate_plan is None:
            self.candidate_plan = build_candidate_plan(
                self.configs, self.runtime_block_arg_names
            )
        autotune_start_time = time.perf_counter()
        if len(self.launchers) == 0:
            self._costmodel_runtime_args = args
            self._costmodel_runtime_kwargs = kwargs
            start_time = time.time_ns()
            self.precompile()
            self.precompile_time_taken_ns = time.time_ns() - start_time

        candidate_count = len(self.candidate_plan["candidate_entries"])
        if not self.variant_launcher_map:
            self._refresh_variant_launchers()

        if candidate_count == 1:
            entry = self._first_compiled_candidate_entry()
            candidate = entry["candidate"]
            launcher = entry["launcher"]
            self._set_best_candidate(candidate, launcher)
            self.launchers = [launcher]
            log.info(f"{self.get_fn_name()} benchmark elapsed time {time.perf_counter() - autotune_start_time}s")
            return

        if self._should_skip_autotune_for_determinism():
            entry = self._first_compiled_candidate_entry()
            candidate = entry["candidate"]
            launcher = entry["launcher"]
            self._set_best_candidate(candidate, launcher)
            self.launchers = [launcher]
            if self.save_cache_hook:
                best_config = (
                    self.best_candidate_config
                    if self.best_candidate_config is not None
                    else self.launchers[0].config
                )
                self.save_cache_hook(best_config, 0)
        else:
            self.autotune_to_one_config(*args, **kwargs)
        log.info(f"{self.get_fn_name()} benchmark elapsed time {time.perf_counter() - autotune_start_time}s")

    def _interpret_args_grid(
            self, args: tuple[Any, ...], cfg: Config, runtime_blocks=None
    ) -> tuple[tuple[Any, ...], tuple[int, int, int]]:

        numels = [
            arg
            for arg in self.fn.arg_names
            if "_numel" in arg
        ]
        runtime_blocks = runtime_blocks or ()
        if self.inductor_meta.get("group_enabled", False):
            runtime_block_names = tuple(
                self.inductor_meta.get("runtime_block_arg_names", ())
            )
            grid = GridExprNpu.from_grouped_meta_and_numel(
                self.inductor_meta,
                cfg,
                numels,
                runtime_block_names=runtime_block_names,
            ).eval_slow(
                dict(
                    zip(
                        [
                            *self.fn.arg_names,
                            *self.inductor_meta.get("extra_launcher_args", ()),
                            *runtime_block_names,
                        ],
                        (*args, *runtime_blocks),
                    )
                )
            )
        elif self.runtime_block_arg_names:
            grid = GridExprNpu.from_meta_and_set_numel(
                self.inductor_meta, cfg, numels
            ).eval_slow(
                dict(
                    zip(
                        [
                            *self.fn.arg_names,
                            *self.inductor_meta.get("extra_launcher_args", ()),
                            *self.runtime_block_arg_names,
                        ],
                        (*args, *runtime_blocks),
                    )
                )
            )
        else:
            grid = GridExprNpu.from_meta_and_set_numel(self.inductor_meta, cfg, numels).eval_slow(
                dict(
                    zip(
                        [
                            *self.fn.arg_names,
                            *self.inductor_meta.get("extra_launcher_args", ()),
                        ],
                        args,
                    )
                )
            )
        if self.inductor_meta.get("extra_launcher_args"):
            args = args[: -len(self.inductor_meta["extra_launcher_args"])]
        return args, grid


class NPUSymbolicGroupedAutotuner(NPUCachingAutotuner):
    def __init__(
            self,
            fn,
            triton_meta,
            configs,
            save_cache_hook,
            mutated_arg_names: List[str],
            optimize_mem,
            heuristic_type,
            size_hints=None,
            inductor_meta=None,
            custom_kernel=False,
            filename: Optional[str] = None,
            reset_to_zero_arg_names: Optional[List[str]] = None,
    ):
        super().__init__(
            fn,
            triton_meta,
            configs,
            save_cache_hook,
            mutated_arg_names,
            optimize_mem,
            heuristic_type,
            size_hints,
            inductor_meta,
            custom_kernel,
            filename,
            reset_to_zero_arg_names,
        )
        self.candidate_plan = self.inductor_meta["grouped_candidate_plan"]
        self.reachable_selection_keys = tuple(
            self.candidate_plan.get(
                "reachable_group_ids",
                range(self.candidate_plan["group_id_count"]),
            )
        )
        self.best_candidate_map = {}
        self.best_launcher_map = {}
        self._grouped_runtime_args_snapshot = ()
        self._grouped_variant_launchers_initialized = False

    def _set_group_best_candidate(self, group_id, candidate, launcher):
        self.best_candidate_map[group_id] = candidate
        self.best_launcher_map[group_id] = launcher

    def _precompile_variants(self):
        if self._grouped_variant_launchers_initialized:
            return
        if len(self.launchers) == 0:
            start_time = time.time_ns()
            self.kernel_name = self.get_fn_name()
            if getattr(self, "_precompile_worker_parallel", None) is not None:
                self._precompile_worker_parallel()
            else:
                self._precompile_worker()
            self._make_launchers()
            self.precompile_time_taken_ns = time.time_ns() - start_time
        self._refresh_variant_launchers()
        self._grouped_variant_launchers_initialized = True

    def _runtime_feature_inputs(self, args) -> tuple[int, ...]:
        if "feature_arg_indices" not in self.candidate_plan:
            raise RuntimeError("grouped plan is missing feature_arg_indices")
        feature_arg_indices = tuple(self.candidate_plan["feature_arg_indices"])
        feature_sources = tuple(self.candidate_plan.get("feature_sources", ()))
        if feature_sources and len(feature_sources) != len(feature_arg_indices):
            raise RuntimeError("feature_sources and feature_arg_indices must match")
        runtime_feature_inputs = []
        for idxs, feature_source in zip(
            feature_arg_indices,
            feature_sources or ({},) * len(feature_arg_indices),
        ):
            values = tuple(int(args[idx]) for idx in idxs)
            source = feature_source.get("source")
            if source in ("outer_product", "reduction_product"):
                product = 1
                for value in values:
                    product *= value
                runtime_feature_inputs.append(product)
            else:
                if len(values) != 1:
                    raise RuntimeError(
                        f"feature source {source} expects one axis, got {values}"
                    )
                runtime_feature_inputs.append(values[0])
        return tuple(runtime_feature_inputs)

    def _resolve_group_id(self, runtime_feature_inputs: tuple[int, ...]) -> int:
        feature_specs = tuple(self.candidate_plan.get("group_features", ()))
        if len(runtime_feature_inputs) != len(feature_specs):
            raise RuntimeError(
                "runtime_feature_inputs and group_features must have the same length"
            )
        group_id = 0
        stride = 1
        for value, spec in zip(runtime_feature_inputs, feature_specs):
            buckets = tuple(spec["buckets"])
            group_id += bucketize(value, buckets) * stride
            stride *= len(buckets) + 1
        group_id_count = self.candidate_plan.get("group_id_count")
        if group_id_count is not None and group_id >= group_id_count:
            raise RuntimeError(
                f"group_id {group_id} is out of range for group_id_count {group_id_count}"
            )
        return group_id

    def _materialize_runtime_blocks(self, candidate, args) -> tuple[int, ...]:
        runtime_block_names = tuple(self.candidate_plan.get("runtime_block_append_order", ()))
        policy = self.candidate_plan["policies"][candidate["policy_id"]]
        runtime_block_rules = tuple(policy.get("runtime_block_rules", ()))
        primary_group_axis = self.inductor_meta.get("primary_group_axis")
        if primary_group_axis is not None:
            primary_block_name = f"{primary_group_axis.upper()}BLOCK"
            if len(runtime_block_rules) > 1:
                raise RuntimeError(
                    "grouped autotune V1 only supports one runtime block rule"
                )
            if runtime_block_rules and runtime_block_rules[0][0] != primary_block_name:
                raise RuntimeError(
                    "grouped autotune V1 runtime block rule must target primary block "
                    f"{primary_block_name}"
                )
        static_blocks = dict(policy.get("static_blocks", ()))
        resolved_blocks = dict(static_blocks)
        axis_arg_indices = dict(self.candidate_plan.get("axis_arg_indices", {}))
        grid_target = int(policy["grid_target"])
        for block_name, rule_items in runtime_block_rules:
            rule = dict(rule_items)
            if rule.get("op") != "ceildiv":
                raise RuntimeError(
                    f"runtime block rule op {rule.get('op')} is not supported"
                )
            axis_name = rule["axis_name"]
            if axis_name not in axis_arg_indices:
                raise RuntimeError(f"axis_arg_indices is missing axis {axis_name}")
            axis_numel = int(args[axis_arg_indices[axis_name]])
            resolved_blocks[block_name] = (axis_numel + grid_target - 1) // grid_target
        missing_runtime_block_names = [
            name for name in runtime_block_names if name not in resolved_blocks
        ]
        if missing_runtime_block_names:
            raise RuntimeError(
                "runtime blocks are missing values for "
                f"{missing_runtime_block_names}"
            )
        return tuple(resolved_blocks[name] for name in runtime_block_names)

    def _benchmark_feature_inputs_for_group(self, group_id: int) -> tuple[int, ...]:
        if "benchmark_feature_inputs_by_group" not in self.candidate_plan:
            raise RuntimeError(
                "grouped plan is missing benchmark_feature_inputs_by_group"
            )
        benchmark_inputs = self.candidate_plan["benchmark_feature_inputs_by_group"]
        if group_id >= len(benchmark_inputs):
            raise RuntimeError(
                f"group_id {group_id} is out of range for benchmark_feature_inputs_by_group"
            )
        return tuple(benchmark_inputs[group_id])

    def _materialize_benchmark_args(self, group_id: int):
        if "benchmark_axis_values_by_group" not in self.candidate_plan:
            raise RuntimeError(
                "grouped plan is missing benchmark_axis_values_by_group"
            )

        benchmark_axis_values = self.candidate_plan["benchmark_axis_values_by_group"]
        if group_id >= len(benchmark_axis_values):
            raise RuntimeError(
                f"group_id {group_id} is out of range for benchmark_axis_values_by_group"
            )

        axis_env = {
            axis_name: int(axis_value)
            for axis_name, axis_value in benchmark_axis_values[group_id]
        }

        def eval_expr(expr):
            if isinstance(expr, int):
                return expr
            if not isinstance(expr, dict):
                raise RuntimeError(f"unsupported benchmark expr payload: {expr}")
            if "const" in expr:
                return int(expr["const"])
            if "runtime_arg_index" in expr:
                runtime_arg_index = int(expr["runtime_arg_index"])
                if runtime_arg_index >= len(self._grouped_runtime_args_snapshot):
                    raise RuntimeError(
                        "benchmark expr runtime_arg_index is out of range: "
                        f"{runtime_arg_index}"
                    )
                return self._grouped_runtime_args_snapshot[runtime_arg_index]
            if "axis_name" in expr:
                axis_name = expr["axis_name"]
                if axis_name not in axis_env:
                    raise RuntimeError(
                        f"benchmark axis env is missing axis {axis_name}"
                    )
                return axis_env[axis_name]
            if "mul" in expr:
                product = 1
                for operand in expr["mul"]:
                    product *= eval_expr(operand)
                return product
            if "add" in expr:
                total = 0
                for operand in expr["add"]:
                    total += eval_expr(operand)
                return total
            if "floordiv" in expr:
                operands = tuple(eval_expr(operand) for operand in expr["floordiv"])
                if len(operands) != 2:
                    raise RuntimeError(
                        f"floordiv benchmark expr expects 2 operands: {expr}"
                    )
                return operands[0] // operands[1]
            raise RuntimeError(f"unsupported benchmark expr payload: {expr}")

        def materialize_dtype(dtype):
            if isinstance(dtype, str):
                dtype_name = dtype.removeprefix("torch.")
                return getattr(torch, dtype_name)
            return dtype

        def workspace_factory(count, device, dtype, zero_mode):
            if hasattr(self, "_benchmark_workspace_factory"):
                return self._benchmark_workspace_factory(count, device, dtype, zero_mode)
            if zero_mode == "ZERO_ON_CALL":
                return torch.zeros(count, device=device, dtype=dtype)
            return torch.empty(count, device=device, dtype=dtype)

        def materialize_spec(spec, kind):
            spec_kind = spec.get("kind")
            source = spec.get("source")
            if source == "runtime_arg":
                runtime_arg_index = int(spec["index"])
                if runtime_arg_index >= len(self._grouped_runtime_args_snapshot):
                    raise RuntimeError(
                        f"{kind} runtime_arg index {runtime_arg_index} is out of range"
                    )
                return self._grouped_runtime_args_snapshot[runtime_arg_index]
            if source == "axis_expr":
                return eval_expr(spec["expr"])
            if spec_kind == "tensor" and source in ("buffer", "constant"):
                size = tuple(eval_expr(expr) for expr in spec["size_exprs"])
                stride = tuple(eval_expr(expr) for expr in spec["stride_exprs"])
                dtype = materialize_dtype(spec["dtype"])
                tensor_factory = getattr(self, "_benchmark_tensor_factory", None)
                if tensor_factory is None:
                    tensor_factory = rand_strided
                return tensor_factory(size, stride, dtype, spec["device"])
            if spec_kind == "workspace" and source == "workspace":
                count = eval_expr(spec["count_expr"])
                return workspace_factory(
                    count,
                    spec["device"],
                    materialize_dtype(spec["dtype"]),
                    spec["zero_mode"],
                )
            raise RuntimeError(f"{kind} spec source {source} is not supported")

        return tuple(
            materialize_spec(spec, f"{spec.get('kind', 'unknown')} arg")
            for spec in self.candidate_plan.get("ordered_arg_specs", ())
        )

    def _build_grouped_benchmark_entries(self):
        entries = []
        compiled_launcher_by_candidate_id = {
            entry["candidate"]["candidate_id"]: entry["launcher"]
            for entry in self.compiled_candidate_entries
        }
        for group_id, candidates in enumerate(self.candidate_plan["group_to_candidates"]):
            if not candidates:
                continue
            benchmark_args = self._materialize_benchmark_args(group_id)
            for candidate in candidates:
                launcher = compiled_launcher_by_candidate_id.get(
                    candidate["candidate_id"]
                )
                if launcher is None or not launcher.runnable:
                    continue
                runtime_blocks = self._materialize_runtime_blocks(candidate, benchmark_args)
                launch_args = self._build_runtime_launch_args(
                    benchmark_args, runtime_blocks
                )
                entries.append(
                    {
                        "group_id": group_id,
                        "candidate": candidate,
                        "launcher": launcher,
                        "runtime_blocks": runtime_blocks,
                        "launch_args": launch_args,
                    }
                )
        return tuple(entries)

    def _grouped_kernel_args_prefix(self, launch_args):
        return launch_args[: len(self.fn.arg_names)]

    def _benchmark_grouped_entries(self, entries, **kwargs):
        if not entries:
            return ()
        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())
        kernel_funcs = []
        kernel_entry_indices = []
        timings = [None] * len(entries)

        def kernel_call(entry):
            launcher = entry["launcher"]
            launch_args = entry["launch_args"]
            kernel_args_prefix = self._grouped_kernel_args_prefix(launch_args)

            def call_kernel():
                if not launcher.runnable:
                    return
                if launcher.config.pre_hook is not None:
                    pre_hook_args = {
                        **dict(zip(self.arg_names, launch_args)),
                        **launcher.config.kwargs,
                    }
                    launcher.config.pre_hook(
                        pre_hook_args
                    )
                cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
                self.reset_to_zero_args(*kernel_args_prefix, **kwargs)
                launcher(
                    *cloned_args,
                    **cloned_kwargs,
                    stream=stream,
                )

            return call_kernel

        for entry_idx, entry in enumerate(entries):
            launcher = entry["launcher"]
            n_spills = launcher.n_spills
            if not self.custom_kernel and n_spills > self.inductor_meta.get(
                "spill_threshold", 16
            ):
                timings[entry_idx] = float("inf")
                continue
            kernel_funcs.append(kernel_call(entry))
            kernel_entry_indices.append(entry_idx)

        if not kernel_funcs:
            return tuple(timings)

        batch_timings = self._benchmark_kernel_funcs_batch(kernel_funcs, "grouped")
        if batch_timings is not None:
            for entry_idx, timing in zip(kernel_entry_indices, batch_timings):
                timings[entry_idx] = timing
            return tuple(timings)

        return tuple(
            timing
            if timing is not None
            else self._bench_with_launch_args(
                entry["launcher"],
                entry["launch_args"],
                self._grouped_kernel_args_prefix(entry["launch_args"]),
                **kwargs,
            )
            for entry, timing in zip(entries, timings)
        )

    def _autotune_all_groups(self, *args, **kwargs):
        self._grouped_runtime_args_snapshot = args
        entries = self._build_grouped_benchmark_entries()
        entry_group_ids = {entry["group_id"] for entry in entries}
        missing_entry_groups = tuple(
            group_id
            for group_id in self.reachable_selection_keys
            if group_id not in entry_group_ids
        )
        if missing_entry_groups:
            raise RuntimeError(
                "reachable grouped autotune groups have no benchmark entries after "
                "precompile/make_launcher filtering: "
                f"{missing_entry_groups}"
            )
        timings = self._benchmark_grouped_entries(entries, **kwargs)
        if len(timings) != len(entries):
            raise RuntimeError(
                "grouped benchmark timing count does not match benchmark entries"
            )
        best_by_group = {}
        for entry, timing in zip(entries, timings):
            group_id = entry["group_id"]
            if group_id not in best_by_group or timing < best_by_group[group_id][0]:
                best_by_group[group_id] = (timing, entry)
        missing_best_groups = tuple(
            group_id
            for group_id in self.reachable_selection_keys
            if group_id not in best_by_group
        )
        if missing_best_groups:
            raise RuntimeError(
                "reachable grouped autotune groups failed to select a best candidate: "
                f"{missing_best_groups}"
            )
        for group_id, (_, best_entry) in best_by_group.items():
            self._set_group_best_candidate(
                group_id,
                best_entry["candidate"],
                best_entry["launcher"],
            )

    def _all_reachable_groups_tuned(self):
        return all(
            group_id in self.best_launcher_map
            for group_id in self.reachable_selection_keys
        )

    def ensure_grouped_autotune_ready(self, *args, **kwargs):
        with self.lock:
            if self._all_reachable_groups_tuned():
                return
            self._precompile_variants()
            self._autotune_all_groups(*args, **kwargs)

    def run(
        self, *args, stream, benchmark_run=False, **kwargs
    ):  # type:ignore[override]
        self.ensure_grouped_autotune_ready(*args, **kwargs)
        runtime_feature_inputs = self._runtime_feature_inputs(args)
        group_id = self._resolve_group_id(runtime_feature_inputs)
        # Grouped autotune reuses the same keyed winner caches, indexed by group_id.
        candidate = self.best_candidate_map.get(group_id)
        launcher = self.best_launcher_map.get(group_id)
        if candidate is None or launcher is None:
            raise RuntimeError(
                f"runtime inputs resolved to unreachable grouped autotune group {group_id}"
            )
        runtime_blocks = self._materialize_runtime_blocks(
            candidate,
            args,
        )
        selected_config = candidate.get(
            "full_config",
            self.candidate_plan["variants"][candidate["variant_id"]]["config"],
        )
        log.info(
            "grouped dispatch group: kernel=%s feature_inputs=%s group_id=%s selected_config=%s runtime_blocks=%s",
            self.inductor_meta.get("kernel_name", self.fn.__name__),
            runtime_feature_inputs,
            group_id,
            selected_config,
            runtime_blocks,
        )
        return launcher(
            *self._build_runtime_launch_args(args, runtime_blocks),
            stream=stream,
            **kwargs,
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

        if npu_config.fasta_autotune:
            from .fasta_autotune import NPUFastAutotuner
            return NPUFastAutotuner(
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

        if inductor_meta.get("group_enabled", False):
            return NPUSymbolicGroupedAutotuner(
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


def template(
    num_stages,
    num_warps,
    triton_meta,
    filename=None,
    inductor_meta=None,
    npu_compile_options=None,
):
    """Compile an NPU Triton template with backend-specific options."""
    return cached_autotune(
        None,
        [
            triton.Config(
                dict(npu_compile_options or {}),
                num_stages=num_stages,
                num_warps=num_warps,
            )
        ],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def patch_triton_heuristics_cached_autotune():
    torch._inductor.runtime.triton_heuristics.cached_autotune = cached_autotune


def brutal_prune_tiling_configs_if_fast_run(configs, inductor_meta) -> List[Config]:
    import os
    max_num_str = os.environ.get("FAST_RUN_WITH_MAX_TILING_NUM", "-1")
    try:
        max_num = int(max_num_str)
    except ValueError:
        max_num = -1

    if max_num > 0 and len(configs) > max_num:
        configs = configs[-1 * max_num:]
        logging.debug("[%s], prune tiling configs to [%s]",
                    inductor_meta["kernel_name"],
                    len(configs))
    return configs


def set_reduction_runtime_blocks_to_numel(configs, split_axis, axis_names, size_hints, runtime_block_arg_names):
    runtime_block_arg_names = set(runtime_block_arg_names)
    if not runtime_block_arg_names:
        return

    for cfg in configs:
        for axis in split_axis:
            axis_name = axis_names[axis]
            if not axis_name.startswith("r"):
                continue
            block_name = f"{axis_name.upper()}BLOCK"
            if block_name not in runtime_block_arg_names:
                continue
            cfg.kwargs[block_name] = size_hints[axis]

        if "split_blocks" in cfg.kwargs:
            cfg.kwargs["split_blocks"] = tuple(
                cfg.kwargs[f"{axis_names[axis].upper()}BLOCK"]
                for axis in split_axis
            )


# split:sizeof split, xblock:axis1 length, rblock:axis2 length
def triton_config_npu_index(
    size_hints,
    inductor_meta,
    triton_meta=None,
    is_reduction=False,
    is_persistent_reduction=False,

) -> List[Config]:
    if inductor_meta.get("group_enabled", False):
        if inductor_meta.get("extra_launcher_args"):
            raise RuntimeError(
                "grouped autotune reached runtime config with extra_launcher_args; "
                "this path is not supported for the current autogenerated Triton grouped autotune flow"
            )
        return _triton_config_npu_index_grouped(
            size_hints,
            inductor_meta,
            triton_meta=triton_meta,
            is_reduction=is_reduction,
            is_persistent_reduction=is_persistent_reduction,
        )
    return _triton_config_npu_index_legacy(
        size_hints,
        inductor_meta,
        triton_meta=triton_meta,
        is_reduction=is_reduction,
        is_persistent_reduction=is_persistent_reduction,
    )


def _triton_config_npu_index_grouped(
    size_hints,
    inductor_meta,
    triton_meta=None,
    is_reduction=False,
    is_persistent_reduction=False,
) -> List[Config]:
    group_features = tuple(inductor_meta.get("group_features", ()))
    signature_names = tuple((triton_meta or {}).get("signature", {}).keys())
    axis_names = tuple(inductor_meta.get("axis_names", ()))
    axis_static_values = tuple(inductor_meta.get("axis_static_values", ()))

    axis_arg_indices = {}
    for axis_name in axis_names:
        arg_name = f"{axis_name}_numel"
        if arg_name in signature_names:
            axis_arg_indices[axis_name] = signature_names.index(arg_name)

    def resolve_feature_arg_indices(feature_spec) -> tuple[int, ...]:
        source = feature_spec["source"]
        axis_names = tuple(feature_spec["axis_names"])
        if source in ("primary_axis", "axis") and len(axis_names) != 1:
            raise RuntimeError(
                f"group feature {feature_spec['name']} expects exactly one axis, got {axis_names}"
            )
        if source not in ("primary_axis", "axis", "outer_product", "reduction_product"):
            raise RuntimeError(
                f"group feature source {source} is not supported by feature_arg_indices v1"
            )
        indices = []
        for axis_name in axis_names:
            arg_name = f"{axis_name}_numel"
            if arg_name not in signature_names:
                raise RuntimeError(
                    f"group feature {feature_spec['name']} arg {arg_name} is missing from signature"
                )
            indices.append(signature_names.index(arg_name))
        return tuple(indices)

    feature_arg_indices = tuple(
        resolve_feature_arg_indices(feature_spec) for feature_spec in group_features
    )

    try:
        group_representatives = build_group_representatives(
            group_features,
            axis_names,
            axis_static_values,
        )
    except UnsupportedGroupedPlan as exc:
        raise RuntimeError(
            "invalid grouped autotune plan reached runtime config; "
            "the current grouped codegen guard should have rejected this plan earlier"
        ) from exc
    group_id_count = group_representatives["group_id_count"]
    reachable_group_ids = set(group_representatives["reachable_group_ids"])
    runtime_block_arg_names = tuple(inductor_meta.get("runtime_block_arg_names", ()))
    primary_group_axis = inductor_meta.get("primary_group_axis")
    if primary_group_axis is None:
        raise RuntimeError("grouped autotune plan is missing primary_group_axis")
    primary_block_name = f"{primary_group_axis.upper()}BLOCK"
    if runtime_block_arg_names and primary_block_name not in runtime_block_arg_names:
        raise RuntimeError(
            f"runtime_block_arg_names is missing primary block {primary_block_name}"
        )

    def benchmark_axis_env(group_id: int) -> dict[str, int]:
        axis_values = group_representatives["benchmark_axis_values_by_group"][group_id]
        return {axis_name: int(axis_value) for axis_name, axis_value in axis_values}

    primary_feature_index = find_primary_feature_index(
        group_features, primary_group_axis
    )
    base_size_hints = {
        axis_name: int(size_hints.get(axis_name, 1))
        for axis_name in axis_names
    }

    def representative_size_hints(group_id: int) -> dict[str, int]:
        representative = dict(base_size_hints)
        representative.update(benchmark_axis_env(group_id))
        return representative

    variant_configs = []
    variant_key_to_id = {}
    policies = {}
    group_to_candidates = []
    candidate_entries = []
    for group_id in range(group_id_count):
        if group_id not in reachable_group_ids:
            group_to_candidates.append(())
            continue
        legacy_configs = _triton_config_npu_index_legacy(
            representative_size_hints(group_id),
            inductor_meta,
            triton_meta=triton_meta,
            is_reduction=is_reduction,
            is_persistent_reduction=is_persistent_reduction,
        )
        group_candidates = []
        group_candidate_keys = set()
        for variant_idx, legacy_cfg in enumerate(legacy_configs):
            variant_cfg = strip_runtime_blocks_from_cfg(
                legacy_cfg, runtime_block_arg_names
            )
            variant_key = repr(config_to_dict(variant_cfg))
            variant_id = variant_key_to_id.get(variant_key)
            if variant_id is None:
                variant_id = f"v{len(variant_configs)}"
                variant_key_to_id[variant_key] = variant_id
                variant_configs.append(variant_cfg)
            policy_id = f"p{group_id}_{variant_id}"
            if len(legacy_configs) > 1:
                policy_id = f"{policy_id}_{variant_idx}"
            policy = build_grouped_launch_policy(
                group_id=group_id,
                cfg=legacy_cfg,
                runtime_block_arg_names=runtime_block_arg_names,
                group_features=group_features,
                primary_group_axis=primary_group_axis,
                primary_feature_index=primary_feature_index,
                axis_env=benchmark_axis_env(group_id),
                npu_num_vector_core=npu_config.num_vector_core,
            )
            candidate_key = (
                variant_id,
                tuple(policy.get("static_blocks", ())),
                tuple(policy.get("runtime_block_rules", ())),
                int(policy.get("grid_target", 1)),
            )
            if candidate_key in group_candidate_keys:
                continue
            group_candidate_keys.add(candidate_key)
            policies[policy_id] = policy
            candidate = {
                "candidate_id": f"g{group_id}_c{len(group_candidates)}",
                "selection_key": group_id,
                "variant_id": variant_id,
                "policy_id": policy_id,
                "full_config": config_to_dict(legacy_cfg),
                "runtime_blocks": tuple(policy.get("static_blocks", ())),
            }
            group_candidates.append(candidate)
            candidate_entries.append(candidate)
        group_to_candidates.append(tuple(group_candidates))

    variant_order = tuple(f"v{idx}" for idx in range(len(variant_configs)))
    ordered_arg_specs = tuple(inductor_meta.get("ordered_arg_specs", ()))
    tensor_arg_specs = tuple(inductor_meta.get("tensor_arg_specs", ()))
    size_arg_specs = tuple(inductor_meta.get("size_arg_specs", ()))
    workspace_arg_specs = tuple(inductor_meta.get("workspace_arg_specs", ()))
    extra_launcher_arg_specs = tuple(inductor_meta.get("extra_launcher_arg_specs", ()))
    grouped_candidate_plan = {
        "variants": {
            f"v{idx}": {"config": config_to_dict(cfg)}
            for idx, cfg in enumerate(variant_configs)
        },
        "variant_order": variant_order,
        "candidate_entries": tuple(candidate_entries),
        "group_to_candidates": tuple(group_to_candidates),
        "policies": policies,
        "runtime_block_arg_count": len(runtime_block_arg_names),
        "runtime_block_append_order": runtime_block_arg_names,
        "group_id_count": group_id_count,
        "reachable_group_ids": tuple(group_representatives["reachable_group_ids"]),
        "unreachable_group_ids": tuple(group_representatives["unreachable_group_ids"]),
        "group_features": tuple(group_features),
        "axis_arg_indices": axis_arg_indices,
        "feature_arg_indices": feature_arg_indices,
        "feature_sources": tuple(
            {
                "name": feature_spec["name"],
                "source": feature_spec["source"],
                "axis_names": tuple(feature_spec["axis_names"]),
            }
            for feature_spec in group_features
        ),
        "benchmark_feature_inputs_by_group": tuple(
            group_representatives["benchmark_feature_inputs_by_group"]
        ),
        "benchmark_axis_values_by_group": tuple(
            group_representatives["benchmark_axis_values_by_group"]
        ),
        "ordered_arg_specs": ordered_arg_specs,
        "tensor_arg_specs": tensor_arg_specs,
        "size_arg_specs": size_arg_specs,
        "workspace_arg_specs": workspace_arg_specs,
        "extra_launcher_arg_specs": extra_launcher_arg_specs,
    }
    inductor_meta["grouped_candidate_plan"] = grouped_candidate_plan
    candidate_count_by_group = tuple(
        (group_id, len(candidates))
        for group_id, candidates in enumerate(grouped_candidate_plan["group_to_candidates"])
        if candidates
    )
    log.info(
        "grouped codegen groups: kernel=%s group_id_count=%s reachable_group_count=%s total_candidate_count=%s candidate_count_by_group=%s",
        inductor_meta.get("kernel_name", "triton_"),
        group_id_count,
        len(grouped_candidate_plan["reachable_group_ids"]),
        sum(len(candidates) for candidates in grouped_candidate_plan["group_to_candidates"]),
        candidate_count_by_group,
    )
    return variant_configs


def extract_runtime_blocks_from_cfg(
    cfg,
    runtime_block_arg_names: tuple[str, ...],
) -> dict[str, int]:
    cfg_dict = config_to_dict(cfg)
    if isinstance(cfg_dict, dict) and "kwargs" in cfg_dict:
        kwargs = dict(cfg_dict["kwargs"])
    else:
        kwargs = dict(cfg_dict)
    return {
        block_name: int(kwargs[block_name])
        for block_name in runtime_block_arg_names
        if block_name in kwargs
    }


def filter_launcher_def_args(
    fn_arg_names,
    cfg_dict,
    none_args,
    runtime_block_arg_names: tuple[str, ...],
):
    runtime_block_names = set(runtime_block_arg_names)
    if isinstance(cfg_dict, dict) and "kwargs" in cfg_dict:
        cfg_names = set(cfg_dict["kwargs"])
    else:
        cfg_names = set(cfg_dict)
    return [
        name
        for name in fn_arg_names
        if (name not in cfg_names or name in runtime_block_names) and name not in none_args
    ]


def build_candidate_plan(
    configs,
    runtime_block_arg_names: tuple[str, ...],
):
    runtime_block_arg_names = tuple(runtime_block_arg_names)
    variant_configs = []
    variant_key_to_id = {}
    candidate_entries = []

    for idx, full_cfg in enumerate(configs):
        compile_variant = strip_runtime_blocks_from_cfg(
            full_cfg, runtime_block_arg_names
        )
        runtime_blocks = extract_runtime_blocks_from_cfg(
            full_cfg, runtime_block_arg_names
        )
        variant_key = repr(config_to_dict(compile_variant))
        variant_id = variant_key_to_id.get(variant_key)
        if variant_id is None:
            variant_id = f"v{len(variant_configs)}"
            variant_key_to_id[variant_key] = variant_id
            variant_configs.append(compile_variant)
        candidate_entries.append(
            {
                "candidate_id": f"c{idx}",
                "variant_id": variant_id,
                "full_config": config_to_dict(full_cfg),
                "runtime_blocks": tuple(
                    (name, int(runtime_blocks[name]))
                    for name in runtime_block_arg_names
                    if name in runtime_blocks
                ),
            }
        )

    return {
        "variant_order": tuple(f"v{idx}" for idx in range(len(variant_configs))),
        "variants": {
            f"v{idx}": {"config": config_to_dict(cfg)}
            for idx, cfg in enumerate(variant_configs)
        },
        "candidate_entries": tuple(candidate_entries),
        "runtime_block_append_order": runtime_block_arg_names,
    }


def build_grouped_launch_policy(
    group_id: int,
    cfg,
    runtime_block_arg_names: tuple[str, ...],
    group_features,
    primary_group_axis: str,
    primary_feature_index: int,
    axis_env: dict[str, int],
    npu_num_vector_core: int,
) -> dict[str, object]:
    primary_block_name = f"{primary_group_axis.upper()}BLOCK"
    runtime_blocks = extract_runtime_blocks_from_cfg(cfg, runtime_block_arg_names)
    if primary_block_name not in runtime_blocks:
        if runtime_block_arg_names:
            raise RuntimeError(
                f"legacy grouped config is missing primary block {primary_block_name}"
            )
        return {
            "group_id": group_id,
            "static_blocks": (),
            "runtime_block_rules": (),
            "grid_target": 1,
        }
    tail_group = is_open_bucket_group(
        group_features, primary_feature_index, group_id
    )
    if not tail_group:
        return {
            "group_id": group_id,
            "static_blocks": tuple(
                (block_name, runtime_blocks[block_name])
                for block_name in runtime_block_arg_names
                if block_name in runtime_blocks
            ),
            "runtime_block_rules": (),
            "grid_target": 1,
        }

    static_blocks = []
    prior_programs = 1
    for block_name in runtime_block_arg_names:
        if block_name not in runtime_blocks:
            continue
        axis_name = block_name.removesuffix("BLOCK").lower()
        if axis_name == primary_group_axis:
            continue
        static_block = int(runtime_blocks[block_name])
        static_blocks.append((block_name, static_block))
        if axis_name not in axis_env:
            raise RuntimeError(
                f"benchmark axis env is missing split axis {axis_name}"
            )
        prior_programs *= (int(axis_env[axis_name]) + static_block - 1) // static_block

    return {
        "group_id": group_id,
        "static_blocks": tuple(static_blocks),
        "runtime_block_rules": (
            (
                primary_block_name,
                (("op", "ceildiv"), ("axis_name", primary_group_axis)),
            ),
        ),
        "grid_target": max(1, npu_num_vector_core // prior_programs),
    }


def _triton_config_npu_index_legacy(
    size_hints,
    inductor_meta,
    triton_meta=None,
    is_reduction=False,
    is_persistent_reduction=False,
) -> List[Config]:
    num_warps = 1
    num_stages = 2
    configs = []
    split_axis = inductor_meta["split_axis"]
    tiling_axis = inductor_meta["tiling_axis"]
    no_loop_axis = inductor_meta.get("no_loop_axis", [])
    low_dims = inductor_meta["low_dims"]
    split_axis_dtype = inductor_meta["split_axis_dtype"]
    axis_names = inductor_meta["axis_names"]
    dual_reduction = inductor_meta["dual_reduction"]
    input_signature = triton_meta["signature"]
    input_ptr_num = len(list(filter(lambda k: 'ptr' in k, input_signature))) if triton_meta is not None else 0
    npu_kernel_type = NPUKernelType(inductor_meta.get("npu_kernel_type", "simd"))
    size_hints = [size_hints.get(axis_name, 1) for axis_name in axis_names]

    if npu_config.fasta_autotune:
        from .fasta_autotune import FastATileGenerator
        tile_generator = FastATileGenerator(size_hints, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims,
                                            persistent_reduction=is_persistent_reduction,
                                            dtype=split_axis_dtype,
                                            npu_kernel_type=npu_kernel_type,
                                            input_ptr_num=input_ptr_num, dual_reduction=dual_reduction)
        configs = tile_generator.descend_split_tiling()
    else:
        tile_generator = TileGenerator(size_hints, axis_names, tiling_axis, no_loop_axis, split_axis, low_dims,
                                       persistent_reduction=is_persistent_reduction,
                                       dtype=split_axis_dtype,
                                       npu_kernel_type=npu_kernel_type,
                                       input_ptr_num=input_ptr_num, dual_reduction=dual_reduction)
        if npu_kernel_type == NPUKernelType.SIMD_SIMT_MIX:
            tile_generator.set_kernel_type(NPUKernelType.SIMT_ONLY)
            configs.extend(tile_generator.descend_split_tiling())
            tile_generator.set_kernel_type(NPUKernelType.SIMT_TEMPLATE)
            configs.extend(tile_generator.descend_split_tiling())
        else:
            configs = tile_generator.descend_split_tiling()

    if not configs:
        cfg = {}
        for x in split_axis:
            cfg[f"{axis_names[x].upper()}BLOCK"] = size_hints[x]
        for x in tiling_axis:
            cfg[f"{axis_names[x].upper()}BLOCK_SUB"] = size_hints[x]
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

    inductor_ascend_linear_mode = inductor_meta.get(
        "inductor_ascend_linear_mode", "no_linear"
    )
    if inductor_ascend_linear_mode == "no_linear":
        for tiling_cfg in configs:
            tiling_kwargs = copy.deepcopy(tiling_cfg.kwargs)
            for tiling, tling_value in tiling_kwargs.items():
                if isinstance(tiling, str) and tiling.endswith("SUB"):
                    tiling_cfg.kwargs[tiling.rstrip("_SUB")] = tling_value
                    tiling_cfg.kwargs.pop(tiling)
    elif inductor_ascend_linear_mode == "no_linear_loop":
        for tiling_cfg in configs:
            tiling_kwargs = copy.deepcopy(tiling_cfg.kwargs)
            for tiling, tling_value in tiling_kwargs.items():
                if isinstance(tiling, str) and tiling.endswith(
                        "SUB") and tiling.startswith("R"):
                    tiling_cfg.kwargs[tiling.rstrip("_SUB")] = tling_value

    set_reduction_runtime_blocks_to_numel(configs, split_axis, axis_names, size_hints,
                                          inductor_meta.get("runtime_block_arg_names", ()))

    logging.debug("[%s], generate candidate tiling count: [%s]",
                inductor_meta["kernel_name"],
                len(configs))

    # if fast run, we prune the configs to the last max_num configs
    configs = brutal_prune_tiling_configs_if_fast_run(configs, inductor_meta)
    return configs


def strip_runtime_blocks_from_cfg(
    cfg: Config,
    runtime_block_arg_names: tuple[str, ...],
) -> Config:
    cfg_dict = config_to_dict(cfg)
    if isinstance(cfg_dict, dict) and "kwargs" in cfg_dict:
        normalized = copy.deepcopy(cfg_dict["kwargs"])
        for key in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
            if key in cfg_dict:
                normalized[key] = cfg_dict[key]
    else:
        normalized = copy.deepcopy(cfg_dict)
    for name in runtime_block_arg_names:
        normalized.pop(name, None)
    if runtime_block_arg_names:
        # legacy tiling configs mirror runtime BLOCK values in split_blocks
        normalized.pop("split_blocks", None)
    return config_from_dict(normalized)


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    configs = triton_config_npu_index(size_hints, inductor_meta, triton_meta)

    return cached_autotune(
        size_hints,
        [*configs],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )


def reduction(
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

    configs = triton_config_npu_index(
        size_hints,
        inductor_meta=inductor_meta,
        triton_meta=triton_meta,
        is_reduction=True,
    )

    return cached_autotune(
        size_hints,
        [
            *configs,
        ],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.REDUCTION,
    )


def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint

    configs = triton_config_npu_index(
        size_hints,
        inductor_meta=inductor_meta,
        is_reduction=True,
        triton_meta=triton_meta,
        is_persistent_reduction=True,
    )

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )

def foreach(size_hints, triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta

    configs = triton_config_npu_index(
        size_hints,
        inductor_meta=inductor_meta,
        is_reduction=True,
        triton_meta=triton_meta,
        is_persistent_reduction=True,
    )

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def benchmark_all_configs(self, *args, **kwargs):
    with dynamo_timed("benchmark_all_configs"):
        return self._benchmark_all_configs(*args, **kwargs)


def _measure_prerun_ms(kernel_call_fn):
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_event.record()
    kernel_call_fn()
    end_event.record()
    torch.npu.synchronize()
    return start_event.elapsed_time(end_event)


def _select_prerun_top_candidates(
    prerun_timings,
    min_k=10,
    max_k=200,
    max_total_ms=50.0,
):
    if not prerun_timings:
        return []

    ranked = sorted(prerun_timings, key=lambda item: item[1])
    tunner_min_k = min(min_k, len(ranked))

    selected = {candidate_id for candidate_id, _ in ranked[:tunner_min_k]}
    total_ms = sum(time_cost for _, time_cost in ranked[:tunner_min_k])

    for candidate_id, cost_ms in ranked[tunner_min_k:]:
        if len(selected) >= max_k or total_ms >= max_total_ms:
            break
        selected.add(candidate_id)
        total_ms += cost_ms

    return selected


def _benchmark_all_configs(self, *args, **kwargs):
    if getattr(self, "candidate_plan", None) is None:
        self.candidate_plan = build_candidate_plan(
            self.configs, getattr(self, "runtime_block_arg_names", ())
        )
    return self._benchmark_candidate_entries(*args, **kwargs)


def precompile_parallel(
    self,
    warm_cache_only=False,
    reload_kernel: Optional[Callable[[], CachingAutotuner]] = None,
    static_triton_bundle_key: Optional[str] = None,
):
    if reload_kernel is not None:
        self._reload_kernel = reload_kernel
    start_time = time.perf_counter()
    if hasattr(self, "skip_precompile"):
        if self.skip_precompile:
            return

    runtime_args, runtime_kwargs = self._resolve_costmodel_runtime_inputs()
    self._apply_costmodel_to_configs(*runtime_args, **runtime_kwargs)

    if warm_cache_only:
        self.kernel_name = self.get_fn_name()
        self._precompile_worker_parallel()
        log.info(f"kernel: {self.get_fn_name()} precompile elapsed time: {time.perf_counter() - start_time}s")
        return

    if self.compile_results:
        for result in self.compile_results:
            TritonBundler.put(
                triton_hash_to_path_key(result.kernel.hash),
                self.triton_meta.get("device", 0),
            )
        self._make_launchers()
        self._refresh_variant_launchers()
        return

    self._precompile_worker_parallel()
    self._make_launchers()
    self._refresh_variant_launchers()
    log.info(f"kernel: {self.get_fn_name()} precompile elapsed time: {time.perf_counter() - start_time}s")
