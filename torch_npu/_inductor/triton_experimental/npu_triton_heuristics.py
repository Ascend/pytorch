# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Copyright (c) 2013 the respective contributors
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

# Based on triton_heuristics with NPU-specific heuristics. 2.7.1 migration: imports moved to
# torch._inductor.runtime.triton_heuristics; CachingAutotuner.__init__ gains optimize_mem/
# filename/reset_to_zero_arg_names; _precompile_config returns TritonCompileResult (NPU
# launcher subclassed); cached_autotune uses AutotuneCache; autotune_hints_to_configs &
# disable_pointwise_autotuning need new params; GPUTarget takes 3 args; device is DeviceProperties.

import copy
import functools
import logging
import operator
import os
from . import config as ncfg
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple

import torch
import torch_npu  # noqa: F401

from torch._inductor import config

from torch._inductor.runtime.hints import (
    DeviceProperties,
    HeuristicType,
    ReductionHint,
    TileHint,
)
from .device_props import get_npu_vector_core_count, get_npu_ub_size_bytes
from . import device_props
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    TritonCompileResult,
    autotune_hints_to_configs,
    unique_configs,
    triton_config_reduction,
    hash_configs,
    get_first_attr,
)
from torch._inductor.runtime.autotune_cache import AutotuneCache
from torch._inductor.runtime.triton_compat import (
    ASTSource,
    Config,
    GPUTarget,
)
# 2.13.0: 上游 triton_compat 已移除 cc_warp_size；warp_size 改用 DeviceProperties.warp_size
# 字段（NPU 上为 None → 回退 32），与上游 triton_heuristics GPUTarget 构造一致
from torch._inductor.runtime.runtime_utils import ceildiv, get_max_y_grid, next_power_of_2, triton_hash_to_path_key

import triton

log = logging.getLogger(__name__)


# mspti device-time autotune benchmark. The default event bench (_do_bench_npu) times
# back-to-back host launches; for small kernels host dispatch (~90-220us) exceeds device time
# (~20-40us), so elapsed_time measures HOST rate → rankings meaningless (a 19.9us kernel
# reported as 223us). mspti's KernelMonitor reads true on-device duration via hardware callback
# (19.88us). Needs libmspti.so in LD_PRELOAD + the mspti package, else keeps the event path.
try:
    from mspti import KernelMonitor as _MsptiKernelMonitor
except Exception:
    _MsptiKernelMonitor = None


def _mspti_autotune_enabled():
    """mspti device-time bench is usable iff the package imports AND
    libmspti.so is preloaded (the monitor is a silent no-op otherwise, which
    would yield zero records and a false fallback)."""
    if not ncfg.autotune_mspti:
        return False
    if _MsptiKernelMonitor is None:
        return False
    return "libmspti.so" in os.environ.get("LD_PRELOAD", "")


# Warn-once guard for the event-autotune fallback. The event path times host launches, so
# for small kernels its rankings are unreliable and autotune may pick a slower config. Surface
# that ONCE per process (not per kernel, to avoid flooding) and only when the fallback fires.
_warned_event_fallback = False


def _reason_no_mspti():
    """Why the mspti device-time path is unavailable, for the fallback warning."""
    if not ncfg.autotune_mspti:
        return "NPU_AUTOTUNE_MSPTI=0"
    if _MsptiKernelMonitor is None:
        return "mspti python package not importable"
    if "libmspti.so" not in os.environ.get("LD_PRELOAD", ""):
        return "libmspti.so not in LD_PRELOAD"
    return "mspti device-time bench failed at runtime"


def enable_mspti_autotune():
    """Make torch_npu's mspti autotune-benchmark path usable with no shell setup.

    The mspti path gives per-kernel device time during autotuning (far more
    accurate than wall-clock on tiny NPU kernels) but requires libmspti.so resident
    before KernelMonitor.start(), and torch_npu guards on the LD_PRELOAD string.
    LD_PRELOAD can't be set post-loader, so: ctypes dlopen the lib RTLD_GLOBAL (what
    actually makes the callbacks fire), then append its path to LD_PRELOAD to pass
    the string guard. Path follows the active CANN. Loading is cheap and dormant.
    Best-effort: any failure falls back to the PTA profiler.
    """
    if not ncfg.enable_mspti:
        return
    if "libmspti.so" in os.getenv("LD_PRELOAD", ""):
        return  # already preloaded; nothing to do
    ascend_home = os.getenv("ASCEND_HOME_PATH") or os.getenv("ASCEND_TOOLKIT_HOME")
    if not ascend_home:
        return
    lib_path = os.path.join(ascend_home, "tools", "mspti", "lib64", "libmspti.so")
    if not os.path.exists(lib_path):
        return
    try:
        import ctypes
        # Keep a ref alive for the process lifetime so it is never unloaded.
        global _NPU_LIBMSPTI_HANDLE
        _NPU_LIBMSPTI_HANDLE = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return
    _pre = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = f"{lib_path}:{_pre}" if _pre else lib_path


def _mspti_bench_calls(calls, run_one, warmup, active, device_interface,
                       filter_prefix="triton"):
    """Time each launcher's device kernel via one mspti monitor session.

    ``calls`` is the list of runnable launchers; ``run_one(launcher)`` performs
    exactly one measured launch (clone already bound). Launches run in order and
    the per-stream FIFO guarantees records arrive in launch order, so the flat
    duration list slices cleanly into ``(warmup+active)`` chunks per launcher.
    Only kernels whose name starts with ``filter_prefix`` are counted, dropping
    the ``reset_to_zero`` / ``zero_()`` memset kernels that fire between launches.

    Returns a list of median device-us per launcher, or None if the record
    count does not match expectation (caller then falls back to event timing).
    """
    durations_ns = []

    def callback(data):
        name = getattr(data, "name", None)
        if name and name.startswith(filter_prefix):
            durations_ns.append(data.end - data.start)

    monitor = _MsptiKernelMonitor()
    device_interface.synchronize()
    monitor.start(callback)
    try:
        for launcher in calls:
            for _ in range(warmup + active):
                run_one(launcher)
        device_interface.synchronize()
    finally:
        monitor.stop()

    expected = len(calls) * (warmup + active)
    if len(durations_ns) != expected:
        log.debug("[AUTOTUNE][mspti] record count %s != expected %s; falling back to event timing.",
                  len(durations_ns), expected)
        return None

    per_launcher = []
    for i in range(len(calls)):
        chunk = durations_ns[i * (warmup + active):(i + 1) * (warmup + active)]
        active_chunk = sorted(chunk[warmup:])  # drop warmup, use median
        med_ns = active_chunk[len(active_chunk) // 2] if active_chunk else float("inf")
        # Timings dict is in MILLISECONDS (matches _do_bench_npu / event path).
        per_launcher.append(med_ns / 1e6)
    return per_launcher


# 2.13: upstream removed the module-level `disable_pointwise_autotuning` helper and inlined
# `not inductor_meta.get("autotune_pointwise", True)` at every call site. The fork keeps a
# single overridable gate (__init__ rebinds this name so use_deterministic_algorithms doesn't
# lock every autotuned kernel to its default tile — autotune nondeterminism is timing-only).
# Keep the same name/semantics so the override point survives.
def disable_pointwise_autotuning(inductor_meta):
    return not inductor_meta.get("autotune_pointwise", True)


def _fmt_config(cfg):
    try:
        return (f"kwargs={dict(cfg.kwargs)} warps={getattr(cfg, 'num_warps', None)} "
                f"stages={getattr(cfg, 'num_stages', None)}")
    except Exception:
        return str(cfg)




class NPUTritonCompileResult(TritonCompileResult):
    """
    Subclass TritonCompileResult to inject the NPU-specific launcher.
    Uses the standard runner() dispatch path (same as torch_npu) so that
    profilers (e.g. msprof) can capture kernel execution records via launch
    hooks.
    """

    def make_launcher(self):
        from torch._inductor.utils import triton_version_uses_attrs_dict
        from torch._inductor.runtime.triton_heuristics import (
            config_to_dict,
        )

        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn

        binary._init_handles()

        known_constants = {
            arg for i, arg in enumerate(fn.arg_names) if i in fn.constexprs
        }
        none_args = {
            k
            for k, v in compile_meta["constants"].items()
            if v is None and k not in known_constants
        }
        none_args -= set(compile_meta["signature"].keys())

        NPU_CU_COUNT = get_npu_vector_core_count()

        if triton_version_uses_attrs_dict():
            call_args = list(fn.arg_names)
            def_args = list(fn.arg_names)
            if (
                "num_warps" in compile_meta["constants"]
                or "num_stages" in compile_meta["constants"]
            ):
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

        pm = getattr(binary, "packed_metadata", None)
        if pm is not None and not isinstance(pm, dict) and hasattr(pm, "_asdict"):
            binary.packed_metadata = pm._asdict()

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

        # Per-launch host path selection. binary.run is Triton-Ascend's NPULauncher wrapper
        # (its __call__ adds a frame + profiler-flag store per launch, ~1.4us). Two bare C
        # entries from the .so: fast_launch (METH_FASTCALL, no tuple parse/frame, still takes
        # tensors so msprof shape works, declines with -1 under a launch hook → fall back) and
        # launch (METH_VARARGS fallback). TRITON_REGISTER_TENSOR_MSPROF keeps the full wrapper.
        _runner = scope["runner"]
        _msprof_tensor = os.getenv(
            "TRITON_REGISTER_TENSOR_MSPROF", "false"
        ).lower() in ("true", "1")
        _fast = None if _msprof_tensor else getattr(_runner, "fast_launch", None)
        _slow = getattr(_runner, "launch", None)
        if _fast is not None and _slow is not None:
            scope["runner"] = _fast
            scope["slow_runner"] = _slow
            use_fast = True
        elif (not _msprof_tensor) and _slow is not None:
            scope["runner"] = _slow
            use_fast = False
        else:
            use_fast = False  # keep the NPULauncher wrapper (msprof-tensor path)

        # Pointer args: fast-ptr passes the raw int device pointer (arg.data_ptr()) so C
        # getPointer hits its PyLong_Check fast branch, not a tensor.data_ptr() call per tensor
        # per launch (~0.5us/tensor). None-guarded (int 0 == nullptr).
        #
        # NOT safe under msprof profiling: when __MsprofFlagL1 is set the C launch() calls
        # _get_tensor_shape(arg) -> arg.size() on every pointer arg to report shapes. On an int
        # that raises AttributeError, which _get_tensor_shape leaves pending and launch()'s
        # trailing PyErr_Occurred() turns into a hard launch abort ('int' object has no
        # attribute 'size'). The launcher body is generated statically at precompile time, but
        # profiling is toggled at runtime, so a statically-baked int cannot coexist with a
        # runtime-enabled shape report. Default OFF so the launcher always passes tensor objects
        # (the original, profiling-safe path). Set TRITON_FAST_PTR=1 only when you know profiling
        # is disabled for the whole run and want the per-launch attribute-walk saving.
        _fast_ptr = os.getenv("TRITON_FAST_PTR", "false").lower() in ("true", "1")
        if _fast_ptr:
            signature = compile_meta.get("signature", {}) or {}

            def _to_ptr(arg):
                ty = signature.get(arg)
                if isinstance(ty, str) and ty.startswith("*"):
                    return f"({arg}.data_ptr() if {arg} is not None else 0)"
                return arg

            launch_call_args = [_to_ptr(a) for a in call_args]
        else:
            launch_call_args = call_args

        # fast_launch (FASTCALL) expects the launch_metadata-style fixed-arg order (grid_0/1/2,
        # stream, function, metadata, launch_metadata, enter/exit hooks, *kernel_args) —
        # identical to runner_args in the hasattr(binary,"launch_metadata") branch below, so
        # there fast_args is just runner_args. The legacy branch uses a different layout
        # fast_launch rejects, so disable the fast path there and use the slow runner.
        fast_args = None
        if not hasattr(binary, "launch_metadata"):
            if use_fast:
                # Fast entry can't accept this layout; revert to the slow runner.
                use_fast = False
                scope["runner"] = _slow
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
                *launch_call_args,
            ]
        else:
            launch_metadata = (
                f"bin.launch_metadata((grid_0, 1, 1), stream, {', '.join(call_args)})"
                if binary.__class__.launch_enter_hook else "None"
            )
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
                *launch_call_args,
            ]
            # fast_launch takes the same fixed-arg order; reuse it verbatim.
            if use_fast:
                fast_args = list(runner_args)

        if "extra_launcher_args" in self.inductor_meta:
            def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

        xblock_val = cfg.kwargs.get("XBLOCK", 1)
        # Only clamp grid_0 for truly single-node 1D pointwise kernels (npu_num_x_nodes =
        # x-axis node count, incl constexpr). The grid_0 expr below is from xnumel only,
        # correct only for Grid1D: Grid2D/3D self-distribute over program_id(0) but their
        # dominant axis is ynumel (x often a collapsed 512→1 dim), so grid_0 from xnumel
        # under-launches (sigmoid over [batch,1,1,1] → grid_0==1).
        npu_num_x_nodes = self.inductor_meta.get("npu_num_x_nodes", 0)
        grid_type = self.inductor_meta.get("grid_type", "Grid1D")
        is_simple_1d = (npu_num_x_nodes == 1
                        and grid_type == "Grid1D"
                        and "xnumel" in def_args
                        and "R0_BLOCK" not in set(fn.arg_names))

        # A5 (910_95) one-program-per-tile: kernel emits group_size=1/group_base=program_id
        # per free-x shape, so the launcher must launch EXACTLY total_blocks (over/under
        # aliases or drops tiles — odometer periodic modulo total_blocks). Codegen injects a
        # recipe reproducing it host-side (references XBLOCK literal + <x>numel args); over
        # 65535 coreDim folds logical→physical. Falls through to group-dispatch when absent.
        _recipe = self.inductor_meta.get("npu_dispatch_recipe")
        _grid_recipe_lines = None
        if _recipe and device_props.is_a5():
            # Splice every tile-block constexpr the recipe may reference
            # (XBLOCK / YBLOCK / ZBLOCK for Grid1D/2D/3D) as a literal, so the
            # recipe lines exec with the same block sizes the kernel was JIT'd
            # with. Missing one (e.g. YBLOCK on a Grid2D recipe) would NameError
            # in the grid computation and fail every config.
            _rl = [f"    {_bn} = {_bv}" for _bn, _bv in cfg.kwargs.items()
                   if _bn.endswith("BLOCK")]
            _rl += [f"    {ln}" for ln in _recipe["lines"]]
            _tb = " * ".join(_recipe["factors"])
            _rl.append(f"    grid_0 = max(1, {_tb})")
            _grid_recipe_lines = _rl

        if _grid_recipe_lines is not None:
            # Not memoized: total_blocks depends on multiple <x>numel args, and
            # the recipe arithmetic is cheap relative to correctness clarity.
            grid_0_expr = None
            grid_0_is_memoized = False
        elif is_simple_1d:
            # Clamp to NPU_CU_COUNT: below it, the full count wastes overhead on idle cores;
            # the group-dispatch body is well-defined for grid < total_thread (excess lanes
            # get group_size=0). Lower-bound at 1: an unbacked size can be 0 (speech_transformer
            # empty slice → xnumel==0) and CANN rejects coreDim==0 (EE1003); one program is a
            # correct no-op (mask all False). A5 with a free x-axis takes the recipe path.
            grid_0_expr = (
                f"max(1, min((xnumel + {xblock_val} - 1) // {xblock_val}, {NPU_CU_COUNT}))"
            )
            # grid_0 depends only on xnumel (XBLOCK and NPU_CU_COUNT are baked in). It's a
            # constant literal for static shapes and usually stable for dynamic ones, so
            # memoize on the last-seen xnumel with a single-slot cache: the arithmetic +
            # max/min run only when xnumel changes, still correct for any value. The cache
            # cell is bound as a hidden default param (below) so the lookup is LOAD_FAST.
            grid_0_is_memoized = True
        else:
            grid_0_expr = str(NPU_CU_COUNT)
            grid_0_is_memoized = False

        # Per-launch host cost reduction: the launcher body runs on every enqueue, and
        # CPython resolves every free name in runner_args / grid_0_expr via LOAD_GLOBAL. All
        # of them (runner, function, metadata, hooks, num_warps, shared, cta_args, bin, and
        # the max/min builtins) are loop-invariant, so bind them as keyword-default params to
        # compile as LOAD_FAST. Defaults captured once at def time — a pure bytecode win.
        invariant_names = [
            "runner", "slow_runner", "function", "metadata",
            "launch_enter_hook", "launch_exit_hook",
            "num_warps", "shared", "cta_args", "bin",
            "max", "min",
        ]
        scope.setdefault("max", max)
        scope.setdefault("min", min)
        bound = [n for n in invariant_names if n in scope]
        hidden = "".join(f", {n}={n}" for n in bound)
        if _grid_recipe_lines is not None:
            # A5 one-program-per-tile: the injected recipe computes grid_0 as the
            # exact total_blocks (see above). Not memoized -- it depends on
            # multiple <x>numel args, so a single-slot xnumel cache would be wrong.
            grid_lines = _grid_recipe_lines
        elif grid_0_is_memoized:
            # Single-slot grid cache: [last_xnumel, last_grid_0]. Bound as a
            # mutable default so it persists across calls and is read/written via
            # LOAD_FAST. ``-1`` can never equal a real (non-negative) xnumel, so
            # the first call always misses and populates the slot.
            scope["_grid_cache"] = [-1, 0]
            grid_lines = [
                "    if xnumel == _grid_cache[0]:",
                "        grid_0 = _grid_cache[1]",
                "    else:",
                f"        grid_0 = {grid_0_expr}",
                "        _grid_cache[0] = xnumel",
                "        _grid_cache[1] = grid_0",
            ]
            hidden += ", _grid_cache=_grid_cache"
        else:
            grid_lines = [f"    grid_0 = {grid_0_expr}"]
        if use_fast and fast_args is not None:
            # Hot path: call the FASTCALL ``fast_launch``. It returns None on a
            # real launch and a non-None sentinel (-1) only when it declined
            # because a launch hook is present — then route through the slow
            # ``launch`` (bound as ``slow_runner``), which runs the hooks.
            call_lines = [
                f"    if runner({', '.join(fast_args)}) is not None:",
                f"        slow_runner({', '.join(runner_args)})",
            ]
        else:
            call_lines = [
                f"    runner({', '.join(runner_args)})",
            ]
        lines = [
            f"def launcher({', '.join(def_args)}, stream{hidden}):",
            *grid_lines,
            "    grid_1 = 1",
            "    grid_2 = 1",
            *call_lines,
        ]
        exec("\n".join(lines), scope)

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.runnable = True
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        binary_hash = getattr(binary, "hash", None)
        launcher.cache_hash = (
            triton_hash_to_path_key(binary_hash) if binary_hash is not None else None
        )
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # Stash def_args so the autotuner can wrap this launcher with
        # dtype boundary casts (i64->i32, fp64->fp32).
        launcher._npu_def_args = list(def_args)
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
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
        triton_meta,
        configs,
        save_cache_hook,
        mutated_arg_names,
        optimize_mem,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,
        custom_kernel=False,
        filename=None,
        reset_to_zero_arg_names=None,
    ):
        if inductor_meta is not None:
            inductor_meta["coordinate_descent_tuning"] = False
        super().__init__(
            fn,
            triton_meta,
            configs,
            save_cache_hook,
            mutated_arg_names,
            optimize_mem,
            heuristic_type,
            size_hints=size_hints,
            inductor_meta=inductor_meta,
            custom_kernel=custom_kernel,
            filename=filename,
            reset_to_zero_arg_names=reset_to_zero_arg_names,
        )

    def _npu_arg_names(self):
        """Positional argument names for the compiled kernel.

        The base ``CachingAutotuner`` does not expose an ``arg_names``
        attribute; the canonical source is the ordered keys of the triton
        signature (the same source upstream uses for its debug-mode kernel
        kwargs). Fall back to the JITFunction's ``arg_names`` and finally to an
        empty list so the NaN scan degrades to positional labels rather than
        raising.
        """
        names = list(self.triton_meta.get("signature", {}).keys())
        if names:
            return names
        return list(getattr(self.fn, "arg_names", ()) or ())

    def run(self, *args, stream, benchmark_run=False, **kwargs):
        # First call: precompile + autotune
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, **kwargs)

        (launcher,) = self.launchers
        # Once autotune converges to a single launcher the convergence check above is dead
        # weight per launch. Rebind instance run to a closure calling the launcher directly,
        # dropping this autotuner frame (~1.85us/launch). The closure still routes through
        # self so a later launcher swap (recursively_apply_fns after repartition) is picked
        # up: identity check, fall back to the slow path if the launcher set changed.
        if not benchmark_run:
            self.run = self._make_fast_run(launcher)
        return launcher(*args, **kwargs, stream=stream)

    def _make_fast_run(self, launcher):
        """Build the converged fast-path ``run`` bound to a single launcher."""
        launchers = self.launchers

        def fast_run(*args, stream, benchmark_run=False, **kwargs):
            # If the launcher set was rebuilt under us (repartition, cache
            # invalidation), drop back to the full path and re-converge.
            if self.launchers is not launchers or benchmark_run:
                del self.run  # restore the class method
                return self.run(*args, stream=stream,
                                benchmark_run=benchmark_run, **kwargs)
            # Inductor-generated callers always invoke ``kernel.run(*pos,
            # stream=stream0)`` with no extra kwargs, so keep that path free of
            # the empty ``**kwargs`` dict build + unpack. Only the rare caller
            # that actually passes kwargs takes the general branch.
            if kwargs:
                return launcher(*args, **kwargs, stream=stream)
            return launcher(*args, stream=stream)

        return fast_run

    def _make_launchers(self):
        super()._make_launchers()
        downcast_args = self.triton_meta.get("downcast_args")
        if not downcast_args:
            return
        mutated = set(self.mutated_arg_names or ())
        self.launchers = [
            _wrap_launcher_with_downcast(launcher, downcast_args=downcast_args, mutated_arg_names=mutated)
            for launcher in self.launchers
        ]

    def _partition_configs_by_tier(self, configs):
        """Split configs into (primary, fallback) tiers by tile size.

        Primary: large tile (product of all BLOCK kwargs >= PRIMARY_TILE_MIN)
        with reasonable num_warps (>= PRIMARY_WARPS_MIN). These are the
        configs that have a real chance of winning autotune — they cover
        enough work per launch to amortize per-launch overhead and use
        enough warps to keep the vector unit busy.

        Fallback: anything below the threshold (tiny tiles, low num_warps).
        These exist as a defensive safety net but cost compile + bench time
        without ever winning when primary configs are usable. Only compile
        them when every primary config fails (typically UB overflow on
        giant fused kernels).
        """
        PRIMARY_TILE_MIN = 512
        PRIMARY_WARPS_MIN = 4
        primary, fallback = [], []
        for cfg in configs:
            tile = 1
            for k, v in cfg.kwargs.items():
                if k.endswith("BLOCK") and isinstance(v, int) and v > 0:
                    tile *= v
            nwarps = getattr(cfg, "num_warps", 8) or 0
            if tile >= PRIMARY_TILE_MIN and nwarps >= PRIMARY_WARPS_MIN:
                primary.append(cfg)
            else:
                fallback.append(cfg)
        return primary, fallback

    def _compile_configs_batch(self, configs):
        """Compile a batch of configs (parallel where possible).

        Returns (compile_results, compile_failed, last_exc). Compile errors
        are caught per-config so a partial success still produces a usable
        list.
        """
        from triton.compiler.errors import MLIRCompilationError, CompilationError
        from triton.runtime.errors import OutOfResources
        from torch._inductor.async_compile import get_compile_threads

        compile_results = []
        compile_failed: List[Tuple[Config, Exception]] = []
        last_exc = None

        if not configs:
            return compile_results, compile_failed, last_exc

        kernel_name = self.inductor_meta.get("kernel_name", getattr(self.fn, "__name__", "triton_"))
        log.debug("[AUTOTUNE] %s: compiling %s config(s)", kernel_name, len(configs))
        for i, cfg in enumerate(configs):
            log.debug("  [COMPILE CANDIDATE %s] %s", i, _fmt_config(cfg))

        compile_threads = max(1, int(get_compile_threads()))

        if compile_threads <= 1 or len(configs) <= 1:
            for c in configs:
                try:
                    result = self._precompile_config(c)
                    compile_results.append(result)
                    log.debug("  [COMPILE OK] %s", _fmt_config(c))
                except (MLIRCompilationError, CompilationError) as e:
                    log.debug("  [COMPILE FAIL] %s -> %s: %s", _fmt_config(c), type(e).__name__, e)  # noqa: G200
                    log.debug("Skipping config %s due to compilation error: %s", c, e)  # noqa: G200
                    compile_failed.append((c, e))
                    last_exc = e
                except Exception as e:
                    if isinstance(e, OutOfResources):
                        log.debug("  [COMPILE FAIL] %s -> %s: %s", _fmt_config(c), type(e).__name__, e)  # noqa: G200
                        compile_failed.append((c, e))
                        last_exc = e
                    else:
                        raise
            return compile_results, compile_failed, last_exc

        def _compile_one(cfg: Config):
            try:
                return self._precompile_config(cfg), None
            except (MLIRCompilationError, CompilationError, OutOfResources) as e:
                return None, e

        max_workers = min(compile_threads, len(configs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cfg = {executor.submit(_compile_one, c): c for c in configs}
            for future in as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    result, exc = future.result()
                except Exception:
                    raise
                if result is not None:
                    compile_results.append(result)
                    log.debug("  [COMPILE OK] %s", _fmt_config(cfg))
                elif exc is not None:
                    log.debug("  [COMPILE FAIL] %s -> %s: %s", _fmt_config(cfg), type(exc).__name__, exc)
                    log.debug("Skipping config %s due to compilation error: %s", cfg, exc)
                    compile_failed.append((cfg, exc))
                    last_exc = exc
        return compile_results, compile_failed, last_exc

    def _precompile_worker(self):
        """Override to tolerate MLIRCompilationError / CompilationError (e.g. UB overflow).

        Upstream only catches OutOfResources / PTXASError.  On NPU, compilation
        failures such as UB overflow surface as MLIRCompilationError.  We skip
        those configs and only raise if *all* configs fail.
        """
        from torch._inductor.runtime.triton_heuristics import NoTritonConfigsError

        if self.compile_results:
            return

        if self.launchers:
            raise RuntimeError("[triton_experimental] launchers must be empty before precompile")
        if not self.configs:
            raise NoTritonConfigsError("No triton configs are available")

        # Tier 1: compile + bench only the primary (large/reasonable) configs.
        # Small configs are deferred until every primary one fails, so we
        # don't burn compile/bench wall-clock on tiles that have no chance
        # of winning.
        primary, fallback = self._partition_configs_by_tier(self.configs)
        if log.isEnabledFor(logging.DEBUG):
            kernel_name = self.inductor_meta.get("kernel_name", "triton_")
            log.debug("[AUTOTUNE] %s: generated %s config(s), primary=%s, fallback=%s",
                      kernel_name, len(self.configs), len(primary), len(fallback))
            for i, cfg in enumerate(self.configs):
                tier = "primary" if cfg in primary else "fallback"
                log.debug("  [CONFIG %s %s] %s", i, tier, _fmt_config(cfg))
        if not primary:
            # No config met the primary threshold (e.g. tiny size_hints).
            # Treat the whole list as primary so we don't accidentally skip
            # everything.
            primary, fallback = list(self.configs), []

        compile_results, compile_failed, last_exc = self._compile_configs_batch(primary)

        # Tier 2: every primary config failed to compile (typical signature
        # of UB overflow on giant fused kernels). Now allow the small/low-
        # warps configs to compete.
        if not compile_results and fallback:
            log.warning(
                "All %d primary configs failed compile for kernel %s; "
                "trying %d small-tile fallback configs",
                len(primary),
                self.inductor_meta.get("kernel_name", "triton_"),
                len(fallback),
            )
            fb_results, fb_failed, fb_last_exc = self._compile_configs_batch(fallback)
            compile_results = fb_results
            compile_failed.extend(fb_failed)
            last_exc = fb_last_exc or last_exc

        if len(compile_results) == 0:
            fb_results, fb_last_exc = self._compile_small_block_fallback()
            if fb_results:
                log.warning(
                    "All initial configs failed; recovered %d small-block fallback "
                    "config(s) for kernel %s",
                    len(fb_results),
                    self.inductor_meta.get("kernel_name", "triton_"),
                )
                compile_results = fb_results
                last_exc = fb_last_exc or last_exc
            else:
                raise NoTritonConfigsError(
                    f"No valid triton configs. All configs failed compilation "
                    f"(including small-block fallback). "
                    f"Last error: {type(last_exc).__name__}: {last_exc}"
                )
        self.compile_results = compile_results
        self.configs = None

    def _compile_small_block_fallback(self):
        """Try a sweep of small XBLOCK/YBLOCK/ZBLOCK configs when every primary
        config fails to compile (typically UB overflow on giant fused kernels).

        We rebuild Configs from one of the failed configs as a template — that
        preserves auxiliary kwargs like ``auto_blockify_size``
        — but override the *_BLOCK kwargs with powers of two in [1, 256].
        Reduction R*_BLOCK kwargs are left untouched (UB overflow is dominated
        by the pointwise tile; reduction tiles are sized differently).
        """
        from triton.compiler.errors import MLIRCompilationError, CompilationError
        from triton.runtime.errors import OutOfResources

        if not self.configs:
            return [], None

        template = self.configs[0]
        block_keys = [k for k in template.kwargs if k.endswith("BLOCK")
                      and not k.startswith("R")]
        if not block_keys:
            return [], None

        size_hint_for_key = {}
        if self.size_hints:
            for k in block_keys:
                axis = k[:-len("BLOCK")].lower()
                hint = self.size_hints.get(axis)
                if isinstance(hint, int) and hint > 0:
                    size_hint_for_key[k] = hint

        small_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        def _cap(key, val):
            hint = size_hint_for_key.get(key)
            if hint is None:
                return val
            return min(val, max(1, hint))

        seen = set()
        candidates = []
        for primary in small_blocks:
            cfg_kwargs = dict(template.kwargs)
            cfg_kwargs[block_keys[0]] = _cap(block_keys[0], primary)
            for k in block_keys[1:]:
                cfg_kwargs[k] = _cap(k, primary)
            key = tuple(sorted((k, cfg_kwargs[k]) for k in block_keys))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Config(
                cfg_kwargs,
                num_warps=getattr(template, "num_warps", 8),
                num_stages=getattr(template, "num_stages", 1),
            ))

        results = []
        last_exc = None
        for cfg in candidates:
            try:
                results.append(self._precompile_config(cfg))
            except (MLIRCompilationError, CompilationError, OutOfResources) as e:
                last_exc = e
                log.debug("Fallback config %s also failed: %s", cfg, e)  # noqa: G200
        return results, last_exc

    def _precompile_config(self, cfg: Config) -> "NPUTritonCompileResult":
        compile_meta = copy.deepcopy(self.triton_meta)

        cfg_kwargs = cfg.kwargs
        compile_meta["constants"].update(cfg_kwargs)
        for i in self.fn.constexprs:
            arg_name = self.fn.arg_names[i]
            if arg_name not in compile_meta["constants"] and arg_name in (
                "num_warps",
                "num_stages",
            ):
                compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        # triton's opt.debug is used by the ascend backend ONLY to emit dump
        # spew (Dumping intermediate results / precompiled.h / launcher.cxx /
        # [DEBUG] cmd_list); the indirect-indexing assert it normally gates is a
        # CUDA-side injection that NPU's aiv vector kernels don't use. So default
        # it off to keep compiles quiet, and reopen with NPU_TRITON_DEBUG=1.
        compile_meta["debug"] = (
            self.inductor_meta.get("assert_indirect_indexing", True)
            and not self.inductor_meta.get("is_hip", False)
            and ncfg.triton_debug
        )

        # 2.7.1: device_type/cc come from device_props
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        from torch._inductor.runtime import triton_helpers
        triton_helpers.set_driver_to_gpu()

        if not ASTSource:
            raise RuntimeError("Installed triton version too old, please upgrade")

        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ),
        )

        # 2.7.1: GPUTarget takes 3 args; force NPU vector kernel generation
        # 2.13.0: warp_size 用 device_props.warp_size or 32（上游 triton_heuristics 同款惯用法）。
        # 不用 cc_warp_size（2.13.0 release triton_compat 已移除）也不用 warp_size_or_default
        # （2.13.0.dev 旧版 DeviceProperties 无此属性）；warp_size 字段两版都有，NPU 上为 None → 回退 32。
        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            self.device_props.warp_size or 32,
        )

        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            "sanitize_overflow": False,
            # NPU: force vector kernel generation
            "mix_mode": "aiv",
            "enable_bisheng_compile": True,
            # VF Fusion controls (see config.py for documentation)
            "enable_vf_fusion": ncfg.enable_vf_fusion,
            "vf_fusion_mode": ncfg.vf_fusion_mode,
            "vf_merge_level": ncfg.vf_merge_level,
        }
        compile_kwargs = {
            "target": target,
            "options": options,
        }

        try:
            binary = triton.compile(*compile_args, **compile_kwargs)
        except Exception as e:
            from triton.compiler.errors import MLIRCompilationError, CompilationError
            if isinstance(e, (MLIRCompilationError, CompilationError)):
                log.debug(
                    "Triton compilation failed (will skip config): %s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    compile_meta,
                )
            else:
                log.exception(
                    "Triton compilation failed: %s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    compile_meta,
                )
            raise

        # Attach block_hints from inductor_meta for use in NPU grid dispatch
        inductor_meta_with_hints = {
            **self.inductor_meta,
            "block_hints": self.triton_meta.get("block_hints", {}),
        }
        return NPUTritonCompileResult(binary, cfg, compile_meta, inductor_meta_with_hints)

    def _clone_all_args_for_autotune(self, args, kwargs):
        """Deep-copy every tensor arg so autotune never writes the real buffers.

        Unlike upstream maybe_clone_args (which clones only mutated_arg_names),
        this clones ALL tensor args — positional and kwarg — preserving strides.
        Non-tensor args pass through unchanged. The returned copies are what the
        whole autotune (PreRun + event bench) runs against, so no config — not
        even one producing NaN/Inf — can leak into the caller's buffers. The
        converged single launch in run() still uses the real args.
        """
        from torch._inductor.compile_fx import clone_preserve_strides

        def _clone(a):
            if isinstance(a, torch.Tensor):
                return clone_preserve_strides(a)
            return a

        cloned_args = [_clone(a) for a in args]
        cloned_kwargs = {k: _clone(v) for k, v in kwargs.items()}
        return cloned_args, cloned_kwargs

    def benchmark_all_configs(self, *args, **kwargs):
        timings = self._benchmark_all_configs_npu(*args, **kwargs)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("[AUTOTUNE] %s benchmark results:", self.fn.__name__)
            for launcher, t in sorted(timings.items(), key=lambda x: x[1]):
                log.debug("  %s -> %.4f ms", launcher.config, t)
        return timings

    def _benchmark_all_configs_npu(self, *args, **kwargs):
        """Batch benchmark all configs.

        Aligned with torch_npu._inductor.runtime.triton_heuristics._benchmark_all_configs:
        grouped NPU-event timing. No torch_npu.profiler CSV path is used during
        autotune.
        """
        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        # Isolate real buffers from autotune: clone EVERY tensor arg once and run the whole
        # autotune (PreRun + event bench) against the clones. Upstream maybe_clone_args only
        # clones mutated_arg_names, so any buffer a kernel writes that inductor didn't mark
        # mutated (or a non-finite intermediate from a bad config) would leak into the
        # caller's real buffer. Full clone keeps real args untouched; only run() writes them.
        bench_args, bench_kwargs = self._clone_all_args_for_autotune(args, kwargs)

        tilling_kernel_list = []

        def kernel_call(launcher):
            def call_kernel():
                if not launcher.runnable:
                    return
                if launcher.config.pre_hook is not None:
                    launcher.config.pre_hook(
                        {**dict(zip(self._npu_arg_names(), bench_args)), **launcher.config.kwargs}
                    )
                cloned_args, cloned_kwargs = self.clone_args(*bench_args, **bench_kwargs)
                self.reset_to_zero_args(*bench_args, **bench_kwargs)
                launcher(*cloned_args, **cloned_kwargs, stream=stream)
            return call_kernel

        for idx, launcher in enumerate(self.launchers):
            if (
                not self.custom_kernel
                and launcher.n_spills is not None
                and launcher.n_spills > self.inductor_meta.get("spill_threshold", 16)
            ):
                launcher.runnable = False
            kernel_call_fn = kernel_call(launcher)
            tilling_kernel_list.append(kernel_call_fn)

            try:
                kernel_call_fn()
                torch.npu.synchronize()
            except Exception as e:
                launcher.runnable = False
                log.debug("[AUTOTUNE] PreRun [%s] index=%s tiling [%s] failed: %s",  # noqa: G200
                          self.fn.__name__, idx, launcher.config, e)

        valid_tiling_length = len([l for l in self.launchers if l.runnable])
        if not valid_tiling_length:
            raise RuntimeError(f"All tiling for [{self.fn.__name__}] are not runnable.")

        def do_event_benchmark(tilling_kernel_list):
            """Per-launcher grouped NPU-event timing.

            This opens no profiler and needs no LD_PRELOAD. _do_bench_npu times
            groups of repeated kernel
            launches with one event pair and divides by the inner repeat count,
            which amortizes event-record/launch bookkeeping overhead for small
            (~10-100us) kernels.
            """
            out = []
            for idx, launcher in enumerate(self.launchers):
                if not launcher.runnable:
                    out.append(float('inf'))
                    continue
                try:
                    if launcher.config.pre_hook is not None:
                        launcher.config.pre_hook(
                            {**dict(zip(self._npu_arg_names(), bench_args)), **launcher.config.kwargs}
                        )
                    # Clone once outside the measured event window. The old
                    # event path timed kernel_call(), which cloned args and ran
                    # reset_to_zero_args on every measured iteration; for small
                    # kernels that dominated the event timing and skewed config
                    # ranking. The direct call below measures just launcher().
                    cloned_args, cloned_kwargs = self.clone_args(*bench_args, **bench_kwargs)

                    def pre_group():
                        self.reset_to_zero_args(*bench_args, **bench_kwargs)

                    def direct_call():
                        launcher(*cloned_args, **cloned_kwargs, stream=stream)

                    out.append(self._do_bench_npu(direct_call, device_interface, pre_group=pre_group))
                except Exception as e:
                    log.debug("[AUTOTUNE] event bench failed for %s idx=%s cfg=%s: %s",  # noqa: G200
                              self.fn.__name__, idx, launcher.config, e)
                    out.append(float('inf'))
            return out

        def do_mspti_benchmark():
            """Per-launcher TRUE device-time via one mspti monitor session.

            Clones args per-launcher and frees before the next (never holds N
            cloned buffer-sets → no autotune OOM). reset_to_zero runs per launch
            for correctness; its memset kernel is non-``triton``-named so the
            record filter drops it. Returns ms-per-launcher aligned to
            self.launchers, or None to signal a clean fallback to events.
            """
            runnable = [l for l in self.launchers if l.runnable]
            if not runnable:
                return None

            # Bind one measured launch per launcher: clone once (outside the
            # measured window is impossible across launchers sharing a session,
            # so clone up front per launcher but free eagerly after its chunk).
            bound = {}
            for launcher in runnable:
                if launcher.config.pre_hook is not None:
                    launcher.config.pre_hook(
                        {**dict(zip(self._npu_arg_names(), bench_args)), **launcher.config.kwargs}
                    )
                cloned_args, cloned_kwargs = self.clone_args(*bench_args, **bench_kwargs)
                bound[launcher] = (cloned_args, cloned_kwargs)

            def run_one(launcher):
                ca, ck = bound[launcher]
                self.reset_to_zero_args(*bench_args, **bench_kwargs)
                launcher(*ca, **ck, stream=stream)

            warmup_iters = ncfg.mspti_warmup
            active_iters = ncfg.mspti_active
            per = _mspti_bench_calls(
                runnable, run_one, warmup_iters, active_iters, device_interface
            )
            # Free clones promptly regardless of outcome.
            bound.clear()
            if per is None:
                return None
            timing_map = dict(zip(runnable, per))
            return [timing_map.get(l, float("inf")) for l in self.launchers]

        try:
            timinglist = None
            if _mspti_autotune_enabled():
                try:
                    timinglist = do_mspti_benchmark()
                except Exception as e:
                    log.debug("[AUTOTUNE][mspti] device-time bench failed for %s: %s; falling back to events.",  # noqa: G200
                              self.fn.__name__, e)
                    timinglist = None

            if timinglist is None:
                # Grouped NPU-event timing. Do NOT use torch_npu.profiler CSV
                # during autotune: nested profiler sessions corrupt outer traces
                # and profiler startup/parse overhead is high.
                global _warned_event_fallback
                if not _warned_event_fallback:
                    _warned_event_fallback = True
                    log.warning(
                        "[AUTOTUNE] mspti device-time bench unavailable (%s); falling "
                        "back to event-based autotune. Event timing is unreliable for "
                        "small NPU kernels (host dispatch cost dwarfs device compute), "
                        "so autotune may pick a suboptimal config.",
                        _reason_no_mspti(),
                    )
                timinglist = do_event_benchmark(tilling_kernel_list)

            if len(timinglist) != len(self.launchers):
                raise RuntimeError("len(timinglist) != len(self.launchers)")
            timings = dict(zip(self.launchers, timinglist))
        except Exception as e:
            log.debug("[AUTOTUNE] batch benchmark failed for %s: %s, falling back to single bench",  # noqa: G200
                      self.fn.__name__, e)
            timings = {
                launcher: self.bench(launcher, *bench_args, **bench_kwargs) if launcher.runnable else float("inf")
                for launcher in self.launchers
            }

        for k, v in timings.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v)

        # Reset only the clones; the real args were never touched. The clones'
        # device memory is reclaimed when bench_args/bench_kwargs go out of scope
        # as this method returns.
        self.reset_to_zero_args(*bench_args, **bench_kwargs)
        return timings

    def bench(self, launcher, *args, with_profiler=False, **kwargs):
        """NPU bench override: use device events for accurate timing instead of torch.cuda events."""
        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
            "spill_threshold", 16
        ):
            return float("inf")

        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        cpu_copies = self.copy_args_to_cpu_if_needed(*args, **kwargs)

        def kernel_call():
            cloned_args, cloned_kwargs = self.maybe_clone_args(cpu_copies, *args, **kwargs)
            self.reset_to_zero_args(*args, **kwargs)
            launcher(*cloned_args, **cloned_kwargs, stream=stream)
            self.restore_args_from_cpu(cpu_copies)

        return self._do_bench_npu(kernel_call, device_interface)

    @staticmethod
    def _do_bench_npu(fn, device_interface, warmup=25, rep=100, pre_group=None):
        """Grouped NPU-event benchmark.

        A single event pair around one tiny kernel includes event-record and
        launch bookkeeping overhead comparable to 10-100us kernels, which makes
        autotune rankings noisy. Time groups of repeated launches with one event
        pair and divide by the inner repeat count; this keeps event overhead to
        sub-us scale for small kernels while still avoiding profiler sessions.
        Returns milliseconds.
        """
        fn()
        device_interface.synchronize()

        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="npu")

        def _time_group(inner_repeats):
            start_event = device_interface.Event(enable_timing=True)
            end_event = device_interface.Event(enable_timing=True)
            cache.zero_()
            if pre_group is not None:
                pre_group()
            start_event.record()
            for _ in range(inner_repeats):
                fn()
            end_event.record()
            device_interface.synchronize()
            return start_event.elapsed_time(end_event) / max(inner_repeats, 1)

        # Estimate one-launch latency, then choose enough repeats so each event
        # window is several ms. Clamp to avoid pathological autotune time for
        # very slow configs while still amortizing event overhead for tiny ones.
        estimate_ms = max(_time_group(10), 1e-6)
        target_group_ms = ncfg.event_bench_target_ms
        max_inner = ncfg.event_bench_max_inner
        inner_repeats = max(1, min(max_inner, int(target_group_ms / estimate_ms)))

        n_warmup_groups = max(1, int(warmup / max(estimate_ms * inner_repeats, 1e-6)))
        n_repeat_groups = max(3, int(rep / max(estimate_ms * inner_repeats, 1e-6)))
        n_repeat_groups = min(n_repeat_groups, ncfg.event_bench_max_groups)

        for _ in range(n_warmup_groups):
            if pre_group is not None:
                pre_group()
            for _ in range(inner_repeats):
                fn()
        device_interface.synchronize()

        times = [_time_group(inner_repeats) for _ in range(n_repeat_groups)]
        times.sort()
        return times[len(times) // 2]



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
    Override of cached_autotune to redirect to NPU autotuner subclass.
    In 2.7.1, uses AutotuneCache instead of manual file caching.
    """
    configs = unique_configs(configs)
    if len(configs) != 1 and not filename:
        raise ValueError("[triton_experimental] cached_autotune requires a filename when multiple configs are given")
    inductor_meta = {} if inductor_meta is None else inductor_meta

    disabled = inductor_meta.get("force_disable_caches", False)

    autotune_cache = None
    if (
        not disabled
        and filename is not None
        and (len(configs) > 1 or inductor_meta.get("coordinate_descent_tuning"))
        and not os.environ.get("TRITON_INTERPRET", "0") == "1"
    ):
        configs_hash = hash_configs(configs)
        autotune_cache = AutotuneCache.create(inductor_meta, filename, configs_hash)
        if autotune_cache and (best_config := autotune_cache.read_best(inductor_meta, configs)):
            configs = [best_config]

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())
    optimize_mem = inductor_meta.pop("optimize_mem", True)

    if "restore_value" in triton_meta:
        mutated_arg_names += triton_meta.pop("restore_value")

    reset_to_zero_arg_names: List[str] = []
    if "reset_to_zero" in triton_meta:
        reset_to_zero_arg_names.extend(triton_meta.pop("reset_to_zero"))

    def decorator(fn):
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    if tconfig.kwargs["XBLOCK"] != 1:
                        raise RuntimeError("[triton_experimental] XBLOCK must be 1 when the kernel has no XBLOCK parameter")
                    tconfig.kwargs.pop("XBLOCK")

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


def npu_triton_config(
    size_hints,
    x,
    y=None,
    z=None,
    num_stages=1,
    num_elements_per_warp=256,
    min_elem_per_thread=0,
    auto_blockify_size=None,
) -> Config:
    cfg = {"XBLOCK": min(x, size_hints["x"])}
    if y:
        cfg["YBLOCK"] = min(y, size_hints["y"])
    if z:
        cfg["ZBLOCK"] = min(z, size_hints["z"])
    if auto_blockify_size is not None:
        cfg["auto_blockify_size"] = auto_blockify_size
    return Config(cfg, num_warps=8, num_stages=num_stages)


autotune_enhance = ncfg.autotune_enhance
npu_group_dispatch = ncfg.group_dispatch
# When set, a static single-tile INNER reduction pins R0_BLOCK=rnumel (one tile
# covers the whole reduction axis, so r0_mask is provably always-true) and the
# codegen drops the now-dead accumulator ``tl.where`` guard. Non-persistent:
# the multi-pass loop structure is kept, only the per-element select is removed.
npu_elide_reduction_where = ncfg.elide_reduction_where
# When TRITON_ALL_BLOCKS_PARALLEL=1 the Ascend backend folds a >65535 grid down
# to the physical core count.  `auto_blockify_size` controls the chunking of
# the leftmost folded dim.  We expose 1/2/4/8 as autotune candidates (matching
# the values triton-ascend ships in its docs) and let the cache pick a winner.
npu_all_blocks_parallel = ncfg.all_blocks_parallel
_NPU_AUTO_BLOCKIFY_CANDIDATES = (1, 2, 4, 8)
_NPU_TOTAL_CORES = get_npu_vector_core_count()

# Lower bound on grid blocks a 1D-pointwise config must produce, as _NPU_TOTAL_CORES //
# this divisor (floored at 2). Fewer blocks than cores is NOT automatically bad: on a small
# memory-bound kernel a larger tile (fewer blocks, some cores idle) can beat full occupancy
# when per-tile overhead dominates. So admit configs down to cores//DIVISOR and let autotune
# bench them; raise the divisor to admit even larger tiles.
_NPU_MIN_BLOCKS_DIVISOR = 4

# At launch, if a tensor is runtime int64 but the compiled signature expects ``*i32`` (NPU
# codegen narrows index dtype), insert a temporary int32 buffer for that arg. For mutated
# args (in/out aliases) the wrapper writes the result back into the original int64 buffer.
# Does NOT correct actual integer overflow inside the kernel — only the ABI dtype mismatch.
npu_int64_boundary_cast = ncfg.int64_boundary_cast

_DTYPE_DOWNCAST_MAP = {
    "*i64": (torch.int64, torch.int32),
    "*fp64": (torch.float64, torch.float32),
}

# When an fp64/i64 arg (RMSNorm eps scalar) feeds more than one kernel, inductor's recompute-
# the-tail fusion drags the SAME source into each, so without memoization every launch re-runs
# the boundary .to() on the AI_CPU scalar engine (~55us per 0-d cast) — N kernels, N casts of
# one value. Memoize the input downcast keyed by source tensor (weak, version-guarded). OUTPUT
# downcasts aren't memoized — each needs a fresh write-back buffer.
npu_dedup_downcast = ncfg.dedup_downcast

# Maps id(src) -> (weakref(src), version, downcast_dtype, downcast_tensor). Keyed by id()
# not the tensor: a dict keyed by tensor would invoke Tensor.__eq__ on lookup — a device-side
# torch.eq (AI_CPU "Equal" op), exactly what this memo avoids. A weakref finalizer drops the
# entry the instant src is freed, so a later tensor reusing the id() can't hit a stale copy;
# the `wref() is src` guard covers the in-flight window too.
_downcast_memo: dict = {}


def _memoized_downcast(src, dst_dtype):
    """Return ``src.to(dst_dtype)``, reusing a cached copy while ``src`` is alive
    and unmutated. Falls back to a plain cast if memoization is disabled or the
    tensor is not weak-referenceable."""
    if not npu_dedup_downcast:
        return src.to(dst_dtype)
    key = id(src)
    try:
        ver = src._version
    except AttributeError:
        return src.to(dst_dtype)
    entry = _downcast_memo.get(key)
    if entry is not None:
        wref, e_ver, e_dtype, e_tmp = entry
        if wref() is src and e_ver == ver and e_dtype == dst_dtype:
            return e_tmp
    tmp = src.to(dst_dtype)
    try:
        wref = weakref.ref(src, lambda _r, _k=key: _downcast_memo.pop(_k, None))
    except TypeError:
        # Not weakly referenceable -- cast without caching.
        return tmp
    _downcast_memo[key] = (wref, ver, dst_dtype, tmp)
    return tmp


def _wrap_launcher_with_downcast(launcher, downcast_args, mutated_arg_names):
    """Wrap a launcher to cast i64/fp64 tensor args to i32/fp32 at the boundary.

    For output args (out_ptr/in_out_ptr or mutated): allocate an empty buffer of
    the downcast dtype, run the kernel, then copy back to the original dtype buffer.
    For input args: cast data to the downcast dtype before the kernel call.
    """
    def_args = getattr(launcher, "_npu_def_args", None)
    if not def_args:
        return launcher

    def_args_tuple = tuple(def_args)
    cast_indices = []
    for i, name in enumerate(def_args_tuple):
        if name in downcast_args:
            orig_type = downcast_args[name]
            if orig_type in _DTYPE_DOWNCAST_MAP:
                src_dtype, dst_dtype = _DTYPE_DOWNCAST_MAP[orig_type]
                is_output = name.startswith("out_ptr") or name.startswith("in_out_ptr") or name in mutated_arg_names
                cast_indices.append((i, name, src_dtype, dst_dtype, is_output))

    if not cast_indices:
        return launcher

    def wrapped(*args, stream=None, **kwargs):
        if len(args) < len(def_args_tuple):
            return launcher(*args, stream=stream, **kwargs)

        new_args = list(args)
        writebacks: List[Tuple[Any, Any, torch.dtype]] = []
        cast_any = False
        for idx, name, src_dtype, dst_dtype, is_output in cast_indices:
            if idx >= len(new_args):
                continue
            a = new_args[idx]
            if not isinstance(a, torch.Tensor) or a.dtype != src_dtype:
                continue
            if is_output:
                tmp = torch.empty_like(a, dtype=dst_dtype)
                writebacks.append((a, tmp, src_dtype))
            else:
                tmp = _memoized_downcast(a, dst_dtype)
            new_args[idx] = tmp
            cast_any = True

        if not cast_any:
            return launcher(*args, stream=stream, **kwargs)

        ret = launcher(*new_args, stream=stream, **kwargs)
        for orig, tmp, src_dt in writebacks:
            orig.copy_(tmp.to(src_dt))
        return ret

    for attr in (
        "config", "runnable", "n_regs", "n_spills", "shared", "store_cubin",
        "fn", "bin", "def_args", "call_args", "_npu_def_args", "cache_hash",
    ):
        if hasattr(launcher, attr):
            setattr(wrapped, attr, getattr(launcher, attr))
    return wrapped


def _filter_balanced_xblock_configs(configs, size_hints, min_xblock_floor=256, axis_hints=None):
    """Drop pointwise configs whose tiling pattern is dispatch-bound.
    1D pointwise: enforce XBLOCK >= max(min_xblock_floor, 256) (with a sane
    fallback when xnumel is smaller) AND total_blocks >= cores //
    _NPU_MIN_BLOCKS_DIVISOR. Tiny XBLOCK on a 1D kernel makes each core walk
    thousands of micro-tiles whose dispatch and control-flow overhead drowns the
    actual vector work. The caller passes its own per-kernel `bs` so very large
    xnumel kernels (where bs > 256) reject mid-range candidates that the
    aligned-block expansion pulls in (e.g. length=8 axis seeds 8,16,32,...,256
    even when xnumel is 13M, where XBLOCK=256 leaves each core walking 1.3K
    tiles). The block-count floor is cores // _NPU_MIN_BLOCKS_DIVISOR rather than
    a hard cores: a large tile with fewer-than-cores blocks can win on a small
    memory-bound kernel where per-tile overhead dominates, so those candidates
    are kept for autotune to bench.

    Multi-axis 1D pointwise (>= 4 constexpr axes): the high-stride floor
    becomes harmful. Each tile is shaped by per-axis ``real_block`` products,
    so XBLOCK grows the tile multiplicatively and large XBLOCK candidates
    overflow UB while small ones are still dispatch-saturated via the axis
    factorisation. In that regime we relax the floor to a constant 256.

    2D/3D pointwise: drop configs whose per-axis BLOCK is below 8 (or below
    the axis numel when the axis itself is smaller). Empirically per-axis
    BLOCKs of 1..4 on a {y, x} kernel pick degenerate tilings (e.g.
    {XBLOCK:32, YBLOCK:4} on a y-fused permute went 12x slower than {32,32}).
    8 is a safe floor that still keeps (8,8)-style square tiles which
    sometimes win on small kernels.
    """
    if not size_hints:
        return list(configs)
    axis_to_kw = {"x": "XBLOCK", "y": "YBLOCK", "z": "ZBLOCK"}

    if len(size_hints) == 1:
        xnumel = size_hints.get("x")
        if not xnumel or xnumel <= 0:
            return list(configs)
        # For small kernels (xnumel < cores * 256), the total_blocks >= cores
        # constraint is impossible to satisfy with any reasonable XBLOCK.
        # In that case, prefer XBLOCK >= xnumel (single block, no loop) or
        # XBLOCK that evenly divides xnumel for balanced dispatch.
        small_kernel = xnumel < _NPU_TOTAL_CORES * 256
        min_xblock = min(max(min_xblock_floor, 256), xnumel)
        kept = []
        for cfg in configs:
            xblock = cfg.kwargs.get("XBLOCK")
            if not isinstance(xblock, int) or xblock <= 0:
                kept.append(cfg)
                continue
            if small_kernel:
                # For small kernels: accept XBLOCK >= xnumel (one block, no
                # loop overhead) or XBLOCK that yields reasonable block count.
                if xblock >= xnumel:
                    kept.append(cfg)
                elif xblock >= min(xnumel, 8):
                    kept.append(cfg)
            else:
                if xblock < min_xblock:
                    continue
                # Upper bound: reject only tiles so large too few blocks remain to keep the
                # machine busy. A larger tile with fewer blocks than cores can still win when
                # per-tile dispatch/loop overhead dominates (measured: a 53K-elem 26-input
                # cat picked XBLOCK=1664 at 32/40 blocks, 27% faster than 1024 at 52/40).
                # So admit down to cores // _NPU_MIN_BLOCKS_DIVISOR and let autotune decide.
                x_blocks = (xnumel + xblock - 1) // xblock
                min_blocks = max(2, _NPU_TOTAL_CORES // _NPU_MIN_BLOCKS_DIVISOR)
                if x_blocks < min_blocks:
                    continue
                kept.append(cfg)
        if not kept:
            return list(configs)
        return kept

    # 2D/3D path: drop configs where any axis BLOCK is too small to amortize
    # dispatch. Empirically a per-axis BLOCK of 1..4 on a {y, x} kernel makes
    # autotune pick degenerate tilings (e.g. {XBLOCK:32, YBLOCK:4} on a y-fused
    # permute went 12x slower than {32,32}). 8 is a safe floor that still keeps
    # (8,8)-style square tiles which sometimes win.
    min_block_2d = 8
    kept = []
    for cfg in configs:
        bad = False
        for axis, numel in size_hints.items():
            kw = axis_to_kw.get(axis)
            if kw is None:
                continue
            block = cfg.kwargs.get(kw)
            if not isinstance(block, int):
                continue
            axis_floor = min(min_block_2d, numel) if isinstance(numel, int) and numel > 0 else min_block_2d
            if block < axis_floor:
                bad = True
                break
        if bad:
            continue
        kept.append(cfg)
    if not kept:
        return list(configs)
    return kept


# Ascend UB (Unified Buffer) capacity in bytes, resolved per-chip: 192 KiB on
# Ascend910B-class parts (the value the bishengir-compile error reports as
# "1572864 bits available"), 256 KiB on Ascend950. Queried from soc_version via
# device_props so a different chip picks up its real capacity instead of a
# hardcoded 910B3 value.
_NPU_UB_CAPACITY_BYTES = get_npu_ub_size_bytes()
# Multi-buffer + scratch overhead factor. The compiler doubles each live
# buffer for software pipelining, and emits scratch tensors for broadcasts.
# 2.0 matches the observed "requires X bits" report on UB-overflow failures
# (e.g. a 256-elem fp32 tile with ~9 live broadcasts reported ~36KB used).
_NPU_UB_OVERHEAD_FACTOR = 2.0
# Cap on how many input tiles the UB estimate treats as simultaneously live. A many-input
# pattern (26-way cat) doesn't hold every tile resident (the compiler streams them), so
# counting all num_load buffers over-estimates peak UB and rejects tiles that fit (XBLOCK
# 1248/1664 predicted to overflow yet ran, 1664 fastest). Overflow is non-fatal (precompile
# worker skips it), so a slight under-estimate only risks a skipped config, never a crash.
_NPU_UB_MAX_CORESIDENT_LOADS = 8

def _estimate_pointwise_tile_bytes(cfg, size_hints, axis_hints, num_load=1, num_reduction=0):
    """Conservative UB-bytes estimate for a 1D-pointwise candidate.

    Tile element count is the product over constexpr axes of
    ``min(length, max(1, XBLOCK // divisor))``. Bytes = tile * 4 (fp32) *
    (num_load + num_reduction + 1 output) * overhead_factor.

    Returns ``None`` when we don't have enough metadata to estimate
    (caller should keep the config in that case).
    """
    if not axis_hints:
        return None
    xblock = cfg.kwargs.get("XBLOCK")
    if not isinstance(xblock, int) or xblock <= 0:
        return None
    tile_elems = 1
    for h in axis_hints:
        length = h.get("length") if isinstance(h, dict) else None
        divisor = h.get("divisor") if isinstance(h, dict) else None
        if not isinstance(length, int) or length <= 0:
            return None
        if not isinstance(divisor, int) or divisor <= 0:
            divisor = 1
        # Mirror the codegen formula: real_block = length if length <= XBLOCK//divisor
        # else max(1, XBLOCK // divisor). When XBLOCK < divisor this collapses
        # the axis to 1, so the tile stays small for that axis.
        per_axis = xblock // divisor if divisor > 0 else xblock
        per_axis = max(1, per_axis)
        per_axis = min(length, per_axis)
        tile_elems *= per_axis
    bytes_per_elem = 4
    live_loads = min(num_load, _NPU_UB_MAX_CORESIDENT_LOADS)
    buffers = max(1, live_loads + num_reduction + 1)
    overhead = _NPU_UB_OVERHEAD_FACTOR
    return int(tile_elems * bytes_per_elem * buffers * overhead)


def _filter_by_ub_estimate(configs, size_hints, axis_hints, inductor_meta):
    """Drop pointwise configs whose estimated tile exceeds NPU UB capacity.

    Only applies to 1D pointwise kernels with constexpr axis hints. Conservative:
    if estimation is impossible we keep the config; if all configs exceed the
    bound we keep the smallest-tile one so autotune still has a candidate
    (it will surface the real compiler error rather than NoTritonConfigsError).
    """
    if not configs or len(size_hints) != 1 or not axis_hints:
        return list(configs)
    inductor_meta = inductor_meta or {}
    num_load = inductor_meta.get("num_load", 1) or 1
    num_reduction = inductor_meta.get("num_reduction", 0) or 0
    annotated = []
    for cfg in configs:
        est = _estimate_pointwise_tile_bytes(
            cfg, size_hints, axis_hints, num_load=num_load, num_reduction=num_reduction
        )
        annotated.append((cfg, est))
    kept = [c for c, est in annotated if est is None or est <= _NPU_UB_CAPACITY_BYTES]
    if kept:
        return kept
    # All candidates over the bound: keep the smallest so the user still sees
    # the compiler's UB-overflow error instead of "No valid triton configs".
    sized = [(c, est) for c, est in annotated if est is not None]
    if not sized:
        return list(configs)
    sized.sort(key=lambda t: t[1])
    return [sized[0][0]]


def _count_x_axis_hints(triton_meta):
    xblock_hints = triton_meta.get("block_hints", {}).get("XBLOCK_HINT", [])
    count = 0
    for hint in xblock_hints:
        if isinstance(hint, dict):
            break
        count += 1
    return count


def _extract_axis_hint_payloads(triton_meta, block_hint_key="XBLOCK_HINT"):
    block_hints = triton_meta.get("block_hints", {}).get(block_hint_key, [])
    static_lengths = [h for h in block_hints if isinstance(h, int) and h > 0]
    divisor_payload = next((h for h in block_hints if isinstance(h, dict) and "divisors" in h), {})
    divisor_hints = [d for d in divisor_payload.get("divisors", []) if isinstance(d, int) and d > 0]
    axis_hints = triton_meta.get("axis_hints", [])
    return static_lengths, divisor_hints, axis_hints


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    if inductor_meta.get("no_x_dim"):
        raise RuntimeError("[triton_experimental] pointwise heuristic does not support no_x_dim")

    numel = functools.reduce(operator.mul, size_hints.values())
    bs = max(256, min(numel // 128, 1024))

    # 2.7.1: autotune_hints_to_configs requires device_props parameter
    device_props: DeviceProperties = triton_meta["device"]
    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", set()), size_hints, bs, device_props
    )

    triton_config_with_settings = functools.partial(
        npu_triton_config, min_elem_per_thread=min_elem_per_thread
    )

    def _filter_reasonable_pointwise_2d_configs(configs):
        return configs

    # 2.7.1: disable_pointwise_autotuning requires inductor_meta parameter
    if len(size_hints) == 1:
        if disable_pointwise_autotuning(inductor_meta) and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, bs)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        elif autotune_enhance:
            import numpy as np
            _, _, axis_hints = _extract_axis_hint_payloads(triton_meta)
            start = bs
            end = 16384
            bs_list = np.floor(np.linspace(start, end, 20))
            bs_list = bs_list // 32 * 32
            bs_list = bs_list.tolist()
            bs_list_base = [bs, bs * 2, bs * 4, bs * 8, bs * 16, bs * 32]

            # In linearize mode, XBLOCK should be a multiple of each static node
            # length so that real_block = node_length // divisor covers as many
            # elements as possible and minimises inner-loop iterations.
            # Extract static node lengths from block_hints and add aligned candidates.
            aligned_bs_list = []
            xblock_hints = triton_meta.get("block_hints", {}).get("XBLOCK_HINT", [])
            static_lengths = [h for h in xblock_hints if isinstance(h, int) and h > 0]
            divisor_payload = next((h for h in xblock_hints if isinstance(h, dict)), {})
            divisor_hints = [d for d in divisor_payload.get("divisors", []) if isinstance(d, int) and d > 0]
            for length in static_lengths:
                k = 1
                while length * k <= 65536:
                    aligned_bs_list.append(length * k)
                    k *= 2
            # For high-stride axes the real bottleneck is often XBLOCK // divisor.
            # Add divisor-aligned large-block candidates so real_block on those
            # axes can exceed 1/2 instead of staying collapsed.
            for divisor in divisor_hints:
                for mul in (1, 2, 4):
                    cand = divisor * mul
                    if cand <= 65536:
                        aligned_bs_list.append(cand)
            aligned_bs_list.append(65536)
            # Always include xnumel itself as a candidate so single-block
            # execution is available for small kernels.
            if numel <= 65536:
                aligned_bs_list.append(numel)
            # Saturation candidates: when xnumel is so large even XBLOCK=65536 leaves each
            # core many mini-tiles, autotune can't pick a one-tile-per-core layout from the
            # seeds. Add geometric candidates between 65536 and numel/total_cores*2 to reach
            # block sizes that saturate the NPU. UB-overflow configs are skipped by
            # _precompile_worker.
            sat_target = max(numel // _NPU_TOTAL_CORES, 65536) * 2
            cand = 65536
            while cand < sat_target and cand < numel:
                cand *= 2
                if cand <= numel:
                    aligned_bs_list.append(int(cand))
            aligned_bs_list = sorted(set(aligned_bs_list))

            def create_triton_configs(size_hints, bs_list, hinted_configs):
                return [
                    triton_config_with_settings(
                        size_hints,
                        bs,
                        num_elements_per_warp=256,
                    )
                    for bs in bs_list_base
                ] + [
                    triton_config_with_settings(
                        size_hints,
                        int(bs),
                        num_elements_per_warp=64,
                    )
                    for bs in bs_list
                ] + [
                    triton_config_with_settings(
                        size_hints,
                        ab,
                    )
                    for ab in aligned_bs_list
                ] + [
                    copy.deepcopy(cfg)
                    for cfg in hinted_configs
                ]

            configs = create_triton_configs(size_hints, bs_list, hinted_configs)
            configs = _filter_balanced_xblock_configs(
                configs, size_hints, min_xblock_floor=bs, axis_hints=axis_hints
            )
            configs = _filter_by_ub_estimate(configs, size_hints, axis_hints, inductor_meta)
            # 3+ axis selector-pattern kernels (select_scatter backward in linearize mode)
            # can have every reasonable XBLOCK overflow UB (the per-axis real_block product
            # blows up the tile). Append an XBLOCK=1 fallback that collapses every axis to
            # real_block=1 so the tile fits UB, even if slow.
            if len(axis_hints) >= 3:
                fallback = triton_config_with_settings(
                    size_hints,
                    128,
                )
                fallback_key = tuple(sorted(fallback.kwargs.items()))
                existing_keys = {tuple(sorted(c.kwargs.items())) for c in configs}
                if fallback_key not in existing_keys:
                    configs.append(fallback)
            return cached_autotune(
                size_hints,
                configs,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        else:
            base_bs = [bs, bs * 2, bs * 4, bs * 8, bs * 16, bs * 32]
            static_lengths, divisor_hints, axis_hints = _extract_axis_hint_payloads(triton_meta)
            extra_bs = [65536]
            for divisor in divisor_hints:
                for mul in (1, 2, 4):
                    cand = divisor * mul
                    if cand <= 65536:
                        extra_bs.append(cand)
            extra_bs = [b for b in sorted(set(extra_bs)) if b not in base_bs]

            # Compile-time pruning strategy:
            # 1) prefer large/divisor-aligned XBLOCK values for high-stride kernels
            # 2) avoid full cartesian products across axes
            all_block_sizes = sorted(set(base_bs + extra_bs))
            preferred_blocks = []
            if all_block_sizes:
                preferred_blocks.append(all_block_sizes[0])
                preferred_blocks.append(all_block_sizes[-1])
            large_divisor_blocks = sorted([b for b in all_block_sizes if b in extra_bs])
            preferred_blocks.extend(large_divisor_blocks[-3:])
            preferred_blocks = sorted(set(preferred_blocks))

            def build_configs(block_sizes):
                return [
                    triton_config_with_settings(size_hints, b)
                    for b in block_sizes
                ]

            if npu_all_blocks_parallel:
                configs_list = build_configs(preferred_blocks)
                large_candidates = preferred_blocks[-2:] if len(preferred_blocks) >= 2 else preferred_blocks
                for b in large_candidates:
                    for ab in _NPU_AUTO_BLOCKIFY_CANDIDATES:
                        if ab == 1:
                            continue
                        configs_list.append(
                            triton_config_with_settings(
                                size_hints,
                                b,
                                auto_blockify_size=ab,
                            )
                        )
            else:
                configs_list = build_configs(preferred_blocks)
            configs_list = _filter_balanced_xblock_configs(
                configs_list + list(hinted_configs), size_hints, min_xblock_floor=bs,
                axis_hints=axis_hints,
            )
            configs_list = _filter_by_ub_estimate(
                configs_list, size_hints, axis_hints, inductor_meta
            )
            return cached_autotune(
                size_hints,
                configs_list,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
    if len(size_hints) == 2:
        _, _, axis_hints = _extract_axis_hint_payloads(triton_meta)
        if (
            disable_pointwise_autotuning(inductor_meta) or tile_hint == TileHint.SQUARE
        ) and not (config.max_autotune or config.max_autotune_pointwise):
            square_configs = [
                triton_config_with_settings(size_hints, 8, 8),
                triton_config_with_settings(size_hints, 16, 16),
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),
                triton_config_with_settings(size_hints, 128, 128),
                triton_config_with_settings(size_hints, 256, 256),
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, 512, 64),
                triton_config_with_settings(size_hints, 64, 512),
                triton_config_with_settings(size_hints, 1024, 32),
                triton_config_with_settings(size_hints, 32, 1024),
                triton_config_with_settings(size_hints, 2048, 16),
                triton_config_with_settings(size_hints, 16, 2048),
                triton_config_with_settings(size_hints, 256, 1024),
                triton_config_with_settings(size_hints, 1024, 256),
                triton_config_with_settings(size_hints, 512, 512),
                triton_config_with_settings(size_hints, 4096, 8),
                triton_config_with_settings(size_hints, 8, 4096),
            ]
            return cached_autotune(
                size_hints,
                _filter_balanced_xblock_configs(
                    _filter_reasonable_pointwise_2d_configs(square_configs), size_hints
                ),
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        base_2d_configs = [
            triton_config_with_settings(size_hints, 8, 8),
            triton_config_with_settings(size_hints, 16, 16),
            triton_config_with_settings(size_hints, 32, 32),
            triton_config_with_settings(size_hints, 64, 64),
            triton_config_with_settings(size_hints, 128, 128),
            triton_config_with_settings(size_hints, 256, 256),
            triton_config_with_settings(size_hints, 256, 16),
            triton_config_with_settings(size_hints, 16, 256),
            triton_config_with_settings(size_hints, 128, 4),
            triton_config_with_settings(size_hints, 4, 128),
            triton_config_with_settings(size_hints, 64, 4),
            triton_config_with_settings(size_hints, 4, 64),
            triton_config_with_settings(size_hints, 32, 4),
            triton_config_with_settings(size_hints, 4, 32),
            triton_config_with_settings(size_hints, bs, 1),
            triton_config_with_settings(size_hints, 1, bs),
            *hinted_configs,
        ]
        # Add size-hint-aware configs: pair the small axis hint (and its
        # divisors) with progressively bigger BLOCKs on the large axis.
        # Filter: only keep configs where XBLOCK*YBLOCK is in [256, 16384]
        # to avoid dispatch-bound (too small) or UB-overflow (too large) tiles.
        y_hint = size_hints.get("y", 1) if isinstance(size_hints, dict) else size_hints[0]
        x_hint = size_hints.get("x", 1) if isinstance(size_hints, dict) else size_hints[1]

        def _divisors_down(n):
            """Yield n, n//2, n//4, ... that evenly divide n, down to 1."""
            divs = []
            v = n
            while v >= 1:
                if n % v == 0:
                    divs.append(v)
                v = v // 2 if v > 1 else 0
            if 1 not in divs:
                divs.append(1)
            return divs

        _TILE_MIN = 256
        # The winning config often lands exactly at _TILE_MAX (256x64), so the tile budget
        # — not the XBLOCK sweep — is the real ceiling. A too-low cap forces large XBLOCK
        # onto tiny YBLOCK (2048x8), starving the parallel axis. Raise the cap (env-tunable)
        # so large XBLOCK pairs with adequate YBLOCK; autotune skips UB-overflow configs.
        _TILE_MAX = ncfg.pointwise_tile_max
        _XB_SWEEP = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        for yb in _divisors_down(y_hint):
            for xb in _XB_SWEEP:
                if xb <= x_hint and _TILE_MIN <= xb * yb <= _TILE_MAX:
                    base_2d_configs.append(
                        triton_config_with_settings(size_hints, xb, yb)
                    )
        for xb in _divisors_down(x_hint):
            for yb in _XB_SWEEP:
                if yb <= y_hint and _TILE_MIN <= xb * yb <= _TILE_MAX:
                    base_2d_configs.append(
                        triton_config_with_settings(size_hints, xb, yb)
                    )
        return cached_autotune(
            size_hints,
            _filter_balanced_xblock_configs(
                _filter_reasonable_pointwise_2d_configs(base_2d_configs), size_hints
            ),
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )

    if len(size_hints) == 3:
        if disable_pointwise_autotuning(inductor_meta):
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, 16, 16, 16)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        configs_3d = [
            triton_config_with_settings(size_hints, 16, 16, 16),
            triton_config_with_settings(size_hints, 64, 8, 8),
            triton_config_with_settings(size_hints, 8, 64, 8),
            triton_config_with_settings(size_hints, 8, 8, 64),
            triton_config_with_settings(size_hints, bs, 1, 1),
            triton_config_with_settings(size_hints, 1, bs, 1),
            triton_config_with_settings(size_hints, 1, 1, bs),
            *hinted_configs,
        ]
        return cached_autotune(
            size_hints,
            configs_3d,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def npu_triton_config_reduction(
    size_hints,
    x,
    r,
    num_stages=1,
    num_warps=None,
    register_intensive=False,
) -> Config:
    # triton_config_reduction requires XBLOCK to divide max_block (4096),
    # so size_hints["x"] must be a power of 2 for the min(x, hint) result
    # to satisfy that constraint.
    aligned_hints = dict(size_hints)
    for key in aligned_hints:
        if isinstance(aligned_hints[key], int) and aligned_hints[key] > 0:
            aligned_hints[key] = next_power_of_2(aligned_hints[key])
    base_cfg = triton_config_reduction(
        aligned_hints,
        x,
        r,
        num_stages=num_stages,
        num_warps=num_warps,
        register_intensive=register_intensive,
    )
    return base_cfg


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
    if inductor_meta.get("no_x_dim"):
        size_hints = dict(size_hints)
        size_hints["x"] = 1

    if triton_meta is None:
        raise ValueError("[triton_experimental] triton_meta must not be None")
    xnumel = size_hints.get("x", 1)
    rnumel = size_hints["r0_"]
    _, divisor_hints, axis_hints = _extract_axis_hint_payloads(triton_meta)
    x_axis_count = _count_x_axis_hints(triton_meta)
    axis_hints = axis_hints[:x_axis_count]

    def reduction_cfg(x, r, *, num_stages=1, num_warps=None, register_intensive=False):
        return npu_triton_config_reduction(
            size_hints,
            x,
            r,
            num_stages=num_stages,
            num_warps=num_warps,
            register_intensive=register_intensive,
        )

    if len(size_hints) == 2:
        contiguous_r = rnumel if 256 <= rnumel <= 16384 else min(rnumel, 16384)
        contiguous_config = reduction_cfg(1, contiguous_r)
        outer_config = reduction_cfg(64, 8)
        tiny_config = reduction_cfg(2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 16384))
        block_hints = triton_meta.get("block_hints", {}) if triton_meta else {}

        def _full_r_range(lo, hi):
            """Generate power-of-2 values from lo to hi inclusive."""
            vals = []
            v = lo
            while v <= hi:
                vals.append(v)
                v *= 2
            return vals

        if config.max_autotune or config.max_autotune_pointwise:
            x_list = [1, 2, 4, 8, 16]
            r_list = _full_r_range(8, 16384)
        elif autotune_enhance:
            x_list = [1, 2, 4, 8, 16]
            r_list = _full_r_range(8, 16384)
        elif reduction_hint == ReductionHint.INNER:
            x_list = [1, 2, 4]
            r_list = _full_r_range(8, contiguous_r)
            if contiguous_r not in r_list:
                r_list.append(contiguous_r)
                r_list.sort()
        elif reduction_hint == ReductionHint.OUTER:
            x_list = [16, 32, 64, 128, 256, 512, 1024]
            r_list = _full_r_range(8, min(rnumel, 256))
        elif reduction_hint == ReductionHint.OUTER_TINY:
            x_list = [2 * (256 // rnumel) if rnumel <= 256 else 1]
            r_list = _full_r_range(8, min(rnumel, 2048))
        elif disable_pointwise_autotuning(inductor_meta):
            x_list = [32]
            r_list = [128]
        else:
            x_list = [1, 2, 4, 8, 16, 32, 64]
            r_list = _full_r_range(8, min(rnumel, 4096))

        for hint_val in block_hints.get("R0_BLOCK_HINT", []):
            if isinstance(hint_val, int) and hint_val > 0 and hint_val not in r_list:
                r_list = sorted(set(r_list) | {hint_val})

        # Always include contiguous_r (the actual rnumel) as a candidate
        if isinstance(contiguous_r, int) and contiguous_r > 0 and contiguous_r not in r_list:
            r_list = sorted(set(r_list) | {contiguous_r})

        # npu_elide_reduction_where: for a static single-reduction-axis kernel fitting one
        # tile, offer two R0_BLOCK candidates: (1) rnumel — r0_mask all-true; (2) align_up
        # (rnumel, 32B) — 32B-aligned lane count (150→152 fp32). Both keep ONE tile so
        # codegen drops the accumulator tl.where: (1) has no pad lanes, (2)'s are neutralised
        # at the LOAD (other=identity). Bounded rnumel<=4096 so the tile fits UB.
        if (
            npu_elide_reduction_where
            and isinstance(rnumel, int)
            and 0 < rnumel <= 4096
        ):
            # Strictest element-count alignment across all pointer operands:
            # the smallest dtype needs the most lanes to reach a 32-byte run,
            # so align to (32 // min_elem_bytes) lanes. Default 4B (fp32).
            def _min_ptr_elem_bytes(sig):
                width = {
                    "fp8": 1, "i8": 1, "u8": 1,
                    "fp16": 2, "bf16": 2, "i16": 2,
                    "fp32": 4, "f32": 4, "i32": 4,
                    "fp64": 4, "f64": 4, "i64": 8,
                }
                best = None
                for v in (sig or {}).values():
                    if isinstance(v, str) and v.startswith("*"):
                        t = v[1:]
                        b = width.get(t)
                        if b is not None:
                            best = b if best is None else min(best, b)
                return best or 4
            _eb = _min_ptr_elem_bytes(
                (triton_meta or {}).get("signature", {})
            )
            _lane_align = max(1, 32 // _eb)
            _aligned = (
                (rnumel + _lane_align - 1) // _lane_align
            ) * _lane_align
            # ADD the single-tile candidates; do NOT overwrite (that drops every smaller
            # R0_BLOCK, so a kernel whose single tile overflows UB — native_layernorm_bw:
            # 400KB tile vs 192KB — has no fitting fallback → all configs fail). Guard
            # elision is sound for ANY R0_BLOCK (pad/tail lanes neutralised at the LOAD,
            # not by assuming one tile), so we only make the aligned tile available.
            _extra = [rnumel]
            if _aligned != rnumel and _aligned <= 4096:
                _extra.append(_aligned)
            r_list = sorted(set(r_list) | set(_extra))

        # XBLOCK_HINT is especially important for multi-x-axis reductions.
        # Convert static per-axis block hints into legal reduction XBLOCK
        # candidates (powers of 2 dividing max_block=4096) so autotune can
        # explore tiles large enough to cover hint-guided x-axis structure.
        xblock_hint_vals = []
        for hint_val in block_hints.get("XBLOCK_HINT", []):
            if isinstance(hint_val, int) and hint_val > 0:
                xblock_hint_vals.append(hint_val)
        if xblock_hint_vals:
            x_hint_product = 1
            for v in xblock_hint_vals:
                x_hint_product *= v
            legal_xblocks = []
            v = 1
            while v <= 4096:
                if v >= min(x_hint_product, 4096):
                    legal_xblocks.append(v)
                v *= 2
            if not legal_xblocks:
                legal_xblocks = [4096]
            x_list = sorted(set(x_list) | set(legal_xblocks))

        if reduction_hint == ReductionHint.DEFAULT:
            if isinstance(xnumel, int) and xnumel > 0:
                x_list = sorted(set(x_list) | {32, 64})
            x_cap = 64
            # Small rnumel ⇒ small RBLOCK ⇒ UB headroom for a larger XBLOCK. The default
            # cap 64 is calibrated for rnumel ~1024+; scale it up when rnumel is small to
            # amortize per-launch overhead. UB budget ≈ XBLOCK*RBLOCK*4B*(loads+reductions+1)
            # *2; targeting ~4K tile elements leaves ~96KB of 192KB for intermediates.
            # Autotuner compile-failure-skip prunes anything that still overflows.
            if isinstance(rnumel, int) and rnumel > 0:
                rblock_p2 = 1 << (rnumel - 1).bit_length() if rnumel > 1 else 1
                rnumel_cap = max(64, 4096 // max(1, rblock_p2))
                x_cap = max(x_cap, min(4096, rnumel_cap))
            # Multi-x-axis reductions: divisor_hints encode tile shape via real_block_xN =
            # XBLOCK // divisor_N. A divisor d>1 needs XBLOCK ≥ d to put >1 element on that
            # axis (XBLOCK < d collapses it to length 1, starving the tile). Lift the cap to
            # several multiples of the largest divisor for aligned candidates above it.
            divisor_max = max((d for d in divisor_hints if isinstance(d, int) and d > 1), default=1)
            if divisor_max > 1:
                x_cap = max(x_cap, min(4096, divisor_max * 8))
            if xblock_hint_vals:
                x_cap = min(4096, max(x_cap, min(x_hint_product, 4096)))
            if isinstance(xnumel, int) and xnumel > 0:
                xnumel_p2 = 1 << (xnumel - 1).bit_length() if xnumel > 1 else 1
                x_cap = min(x_cap, xnumel_p2)
            # Backfill power-of-2 candidates up to the (possibly raised) cap.
            expanded = set(x_list)
            v = 1
            while v <= x_cap:
                expanded.add(v)
                v *= 2
            # Add divisor-aligned candidates so the tile covers the secondary x axis in one
            # launch (d=768 → 768,1536,3072 vs 512/1024 from power-of-2). GEOMETRIC not
            # linear: a small divisor (6=HEADS) with `m += divisor_max` emits x_cap/divisor
            # candidates (~683), ballooning autotune × the R range into thousands of configs.
            # Geometric (d,2d,4d,…) + hard cap 8 bounds the count to O(log(x_cap/divisor)).
            if divisor_max > 1:
                m = divisor_max
                added = 0
                while m <= x_cap and added < 8:
                    expanded.add(m)
                    added += 1
                    m *= 2
            x_list = sorted(x for x in expanded if 0 < x <= x_cap)

        if reduction_hint == ReductionHint.INNER:
            x_upper = min(int(xnumel), 4096) if isinstance(xnumel, int) and xnumel > 0 else 16
            x_candidates = set(x_list)
            v = 1
            while v <= x_upper:
                x_candidates.add(v)
                v *= 2
            x_list = sorted(x for x in x_candidates if x <= x_upper and x > 0) or [1]
        else:
            x_list = sorted({x for x in x_list if x > 0})

        # Unified r_cap: next_power_of_2(rnumel), hard max 65536
        r_cap = min(1 << (rnumel - 1).bit_length(), 65536) if isinstance(rnumel, int) and rnumel > 0 else 65536
        r_list = sorted({r for r in r_list if r > 0 and r <= r_cap})
        if not r_list:
            r_list = [max(8, 1 << (min(int(rnumel) if isinstance(rnumel, int) else 128, 4096) - 1).bit_length())]

        # Thin r_list to ≤6 values: always keep smallest, largest, contiguous_r; sample the
        # rest log-spaced. Also keep the "few-trips" sweet spot: reduction cost is dominated
        # by TRIP COUNT (rnumel // R0_BLOCK) since each trip has fixed overhead, so a narrow
        # R0 is ~1.4x slower than a fat one at the same tile size. Even-index sampling can
        # evict it (rnumel=512 drops R0=64 → T5 2.8x gap), so pin the smallest in-budget R0.
        _MAX_R_CANDIDATES = 6
        _TRIP_BUDGET = 8
        low_trip_r = None
        if isinstance(rnumel, int) and rnumel > _TRIP_BUDGET:
            _target = -(-rnumel // _TRIP_BUDGET)  # ceil(rnumel / budget)
            _target_p2 = 1 << (_target - 1).bit_length() if _target > 1 else 1
            cands = [r for r in r_list if r >= _target_p2]
            if cands:
                low_trip_r = min(cands)
        if len(r_list) > _MAX_R_CANDIDATES:
            must_keep = {r_list[0], r_list[-1]}
            if contiguous_r in r_list:
                must_keep.add(contiguous_r)
            if low_trip_r is not None:
                must_keep.add(low_trip_r)
            remaining_slots = _MAX_R_CANDIDATES - len(must_keep)
            others = [r for r in r_list if r not in must_keep]
            if remaining_slots > 0 and others:
                step = max(1, (len(others) - 1) / (remaining_slots - 1)) if remaining_slots > 1 else len(others)
                sampled = {others[int(round(i * step))] for i in range(remaining_slots) if int(round(i * step)) < len(others)}
            else:
                sampled = set()
            r_list = sorted(must_keep | sampled)

        base_configs = [
            reduction_cfg(x, r)
            for x in x_list
            for r in r_list
        ]

        # TEMP DIAGNOSTIC: pin a single (XBLOCK, R0_BLOCK) to measure the real
        # kernel at a chosen trip count. Remove after validation.
        _pin = ncfg.pin_xr
        if _pin:
            _px, _pr = (int(v) for v in _pin.split(","))
            return cached_autotune(
                size_hints,
                [reduction_cfg(_px, _pr)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )

        if autotune_enhance:
            # The generic enhanced sweep is conservative, leaving a gap that hurts
            # RMSNorm-style backward reductions (INNER: x dynamic rows, r fixed 2048;
            # OUTER: x the 2048 weight axis, r dynamic rows): candidates jump from X<=16 to
            # huge XBLOCK_HINT (2048), missing the medium tiles that balance launch overhead,
            # UB pressure, and occupancy. Add explicit medium pairs; UB checks prune misfits.
            if reduction_hint == ReductionHint.INNER:
                extra_pairs = [
                    (8, 512), (8, 1024),
                    (16, 128), (16, 512), (16, 1024),
                    (32, 64), (32, 128), (32, 256),
                    (64, 64), (64, 128),
                ]
            elif reduction_hint == ReductionHint.OUTER:
                extra_pairs = [
                    (16, 128), (16, 512),
                    (32, 64), (32, 128), (32, 256),
                    (64, 32), (64, 64), (64, 128),
                    (128, 16), (128, 32), (128, 64),
                    (256, 16), (256, 32),
                ]
            else:
                extra_pairs = []
            seen_kwargs = {tuple(sorted(cfg.kwargs.items())) for cfg in base_configs}
            for x, r in extra_pairs:
                if isinstance(xnumel, int) and x > max(1, next_power_of_2(xnumel)):
                    continue
                if isinstance(rnumel, int) and r > max(1, next_power_of_2(rnumel)):
                    continue
                cfg = reduction_cfg(x, r)
                key = tuple(sorted(cfg.kwargs.items()))
                if key not in seen_kwargs:
                    base_configs.append(cfg)
                    seen_kwargs.add(key)

        if not (config.max_autotune or config.max_autotune_pointwise or autotune_enhance):
            preferred_configs = [contiguous_config, outer_config, tiny_config]
            for cfg in preferred_configs:
                if all(cfg.kwargs != existing.kwargs for existing in base_configs):
                    base_configs.append(cfg)

        configs = base_configs

        return cached_autotune(
            size_hints,
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.REDUCTION,
            filename=filename,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """Persistent reduction reuses the reduction autotune config generation
    but emits a PERSISTENT_REDUCTION heuristic type (no loop over R)."""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints = dict(size_hints)
        size_hints["x"] = 1

    if triton_meta is None:
        raise ValueError("[triton_experimental] triton_meta must not be None")
    xnumel = size_hints.get("x", 1)
    rnumel = size_hints["r0_"]
    _, _, axis_hints = _extract_axis_hint_payloads(triton_meta)
    x_axis_count = _count_x_axis_hints(triton_meta)
    axis_hints = axis_hints[:x_axis_count]

    def persistent_cfg(x, r, *, num_stages=1, num_warps=None):
        cfg = npu_triton_config_reduction(
            size_hints,
            x,
            r,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        cfg.kwargs["R0_BLOCK"] = r
        return cfg

    if len(size_hints) == 2:
        block_hints = triton_meta.get("block_hints", {}) if triton_meta else {}

        def _full_r_range(lo, hi):
            vals = []
            v = lo
            while v <= hi:
                vals.append(v)
                v *= 2
            return vals

        if config.max_autotune or config.max_autotune_pointwise:
            x_list = [1, 2, 4, 8, 16]
            r_list = _full_r_range(8, 16384)
        elif autotune_enhance:
            x_list = [1, 2, 4, 8, 16]
            r_list = _full_r_range(8, 16384)
        elif reduction_hint == ReductionHint.INNER:
            contiguous_r = rnumel if 256 <= rnumel <= 16384 else min(rnumel, 16384)
            x_list = [1, 2, 4]
            r_list = _full_r_range(8, contiguous_r)
            if contiguous_r not in r_list:
                r_list.append(contiguous_r)
                r_list.sort()
        elif reduction_hint == ReductionHint.OUTER:
            x_list = [16, 32, 64, 128, 256, 512, 1024]
            r_list = _full_r_range(8, min(rnumel, 256))
        elif reduction_hint == ReductionHint.OUTER_TINY:
            x_list = [2 * (256 // rnumel) if rnumel <= 256 else 1]
            r_list = _full_r_range(8, min(rnumel, 2048))
        elif disable_pointwise_autotuning(inductor_meta):
            x_list = [32]
            r_list = [128]
        else:
            x_list = [1, 2, 4, 8, 16, 32, 64]
            r_list = _full_r_range(8, min(rnumel, 4096))

        # Always include rnumel itself as a candidate (avoids tl.where overhead)
        if isinstance(rnumel, int) and rnumel >= 8 and rnumel not in r_list:
            r_list.append(rnumel)
            r_list.sort()

        for hint_val in block_hints.get("R0_BLOCK_HINT", []):
            if isinstance(hint_val, int) and hint_val > 0 and hint_val not in r_list:
                r_list = sorted(set(r_list) | {hint_val})

        # Persistent reduction: R0_BLOCK = rnumel exactly (no autotune on R).
        # This guarantees r0_mask is always True, eliminating tl.where overhead.
        r_list = [rnumel if isinstance(rnumel, int) and rnumel > 0 else next_power_of_2(rnumel)]

        # Align x_list upper bound with XBLOCK_HINT
        xblock_hint_vals = []
        for hint_val in block_hints.get("XBLOCK_HINT", []):
            if isinstance(hint_val, int) and hint_val > 0:
                xblock_hint_vals.append(hint_val)
        if xblock_hint_vals:
            x_hint_product = 1
            for v in xblock_hint_vals:
                x_hint_product *= v
            # Add powers of 2 up to the hint
            v = 1
            while v <= min(x_hint_product, 4096):
                if v not in x_list:
                    x_list.append(v)
                v *= 2
            x_list = sorted(set(x_list))

        # Cap x_list
        if isinstance(xnumel, int) and xnumel > 0:
            x_list = sorted({x for x in x_list if x > 0 and x <= xnumel})
        if not x_list:
            x_list = [1]

        configs = [
            persistent_cfg(x, r)
            for x in x_list
            for r in r_list
        ]

        # R0_BLOCK is hardcoded in codegen_static_numels, not a constexpr arg
        for c in configs:
            c.kwargs.pop("R0_BLOCK", None)

        if disable_pointwise_autotuning(inductor_meta):
            configs = configs[:1]

        return cached_autotune(
            size_hints,
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
        )
    raise NotImplementedError(f"persistent_reduction size_hints: {size_hints}")


def grid(*numels):
    """
    Helper function to compute triton grids for NPU.
    Used in NPU kernel dispatch: grid_0 = number of AI-vector cores
    (chip-dependent — see _NPU_TOTAL_CORES), grid_1 and grid_2 are
    computed from numels.
    """
    if len(numels) == 1:
        xnumel, ynumel, znumel = numels[0], None, None
    elif len(numels) == 2:
        xnumel, ynumel, znumel = numels[1], numels[0], None
    elif len(numels) == 3:
        xnumel, ynumel, znumel = numels[2], numels[1], numels[0]
    else:
        raise AssertionError(f"invalid size for numels {len(numels)}")

    def get_grid_dim(numel, block, block_hint):
        if numel is None:
            return 1
        if block is None:
            return numel
        if isinstance(block_hint, (list, tuple)) and any(h == -1 for h in block_hint):
            return ceildiv(numel, block)
        if len(block_hint) == 0:
            raise AssertionError("Length of BLOCK_HINT cannot be 0.")
        real_blocks = []
        divisor = 1
        for node_length in block_hint:
            real_blocks.append(
                node_length
                if node_length <= (block // divisor)
                else (block // divisor)
                if (block > divisor)
                else 1
            )
            divisor *= int(node_length)

        total_grid = 1
        for idx in range(len(block_hint)):
            total_grid *= ceildiv(block_hint[idx], real_blocks[idx])
        return total_grid

    max_grid_dims = config.triton.max_tiles

    def grid_fn(meta):
        x_grid = get_grid_dim(
            xnumel, meta.get("XBLOCK", 1), meta.get("XBLOCK_HINT", 1)
        )
        y_grid = get_grid_dim(
            ynumel, meta.get("YBLOCK", None), meta.get("YBLOCK_HINT", 1)
        )

        MAX_Y_GRID = get_max_y_grid()
        if znumel is None and max_grid_dims <= 2:
            div = ceildiv(y_grid, MAX_Y_GRID)
            y_grid = y_grid // div
            z_grid = div
        else:
            z_grid = get_grid_dim(
                znumel, meta.get("ZBLOCK", None), meta.get("ZBLOCK_HINT", 1)
            )
            torch._check(
                y_grid <= MAX_Y_GRID,
                lambda: f"Generated y grid beyond 2^16 ({y_grid}) not supported with z dimension present.",
            )

        return (x_grid, y_grid, z_grid)

    return grid_fn
