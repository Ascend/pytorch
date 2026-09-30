# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

# Mirrored from: pytorch/torch/_inductor/runtime/triton_heuristics.py
# Upstream commit: b62a09d1dd4d337a7d41b73a5c2e38744f9d5ddc
# Local differences: NPU eligibility, npubin recovery and CANN execution.

from __future__ import annotations

import logging
import os
from typing import Any

from torch._dynamo.utils import set_feature_use
from torch._inductor import config
from torch._inductor.runtime.cache_dir_utils import triton_cache_dir
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key
from torch._inductor.runtime.triton_heuristics import CompileResult

from torch_npu import _C

from .. import device_props
from ..device_props import get_npu_vector_core_count
from ..launcher_codegen import _gen_grid_code, _gen_launcher_code
from .adapter import NPUStaticArtifactAdapter, NPUStaticArtifactError
from .kernel import NPUStaticallyLaunchedTritonKernel


log = logging.getLogger("torch._inductor")


class CannotStaticallyLaunchNPUKernel(RuntimeError):
    pass


def _hook_is_empty(hook: Any) -> bool:
    if hook is None:
        return True
    calls = getattr(hook, "calls", None)
    if calls is None:
        return False
    try:
        return len(calls) == 0
    except TypeError:
        return False


class NPUStaticTritonCompileResult(
    CompileResult[NPUStaticallyLaunchedTritonKernel]
):
    @staticmethod
    def can_statically_launch(
        binary: Any,
        *,
        compile_meta: dict[str, Any],
        inductor_meta: dict[str, Any],
        heuristic_type: HeuristicType,
    ) -> NPUStaticallyLaunchedTritonKernel | None:
        if not config.use_static_triton_launcher:
            return None

        def check() -> NPUStaticallyLaunchedTritonKernel:
            launcher = getattr(_C, "_StaticNpuLauncher", None)
            is_supported = getattr(launcher, "_is_supported", None)
            if is_supported is None or not is_supported():
                raise CannotStaticallyLaunchNPUKernel(
                    "CANN runtime or torch_npu build lacks NPU static launcher support "
                    "(requires aclrtLaunchKernelWithHostArgs)"
                )
            if config.cpp_wrapper:
                raise CannotStaticallyLaunchNPUKernel("cpp wrapper enabled")
            if (
                heuristic_type == HeuristicType.USER_AUTOTUNE
                and not config.static_launch_user_defined_triton_kernels
            ):
                raise CannotStaticallyLaunchNPUKernel("user-defined Triton kernel")
            if inductor_meta.get("store_cubin"):
                raise CannotStaticallyLaunchNPUKernel("store_cubin is enabled")
            if os.getenv("TRITON_REGISTER_TENSOR_MSPROF", "false").lower() in (
                "true",
                "1",
            ):
                raise CannotStaticallyLaunchNPUKernel(
                    "Triton tensor profiling is enabled"
                )

            try:
                from torch._inductor.runtime.triton_compat import knobs
            except ImportError:
                knobs = None
            if knobs is None:
                launch_enter = getattr(binary.__class__, "launch_enter_hook", None)
                launch_exit = getattr(binary.__class__, "launch_exit_hook", None)
            else:
                launch_enter = knobs.runtime.launch_enter_hook
                launch_exit = knobs.runtime.launch_exit_hook
            if not _hook_is_empty(launch_enter):
                raise CannotStaticallyLaunchNPUKernel("launch enter hook enabled")
            if not _hook_is_empty(launch_exit):
                raise CannotStaticallyLaunchNPUKernel("launch exit hook enabled")

            try:
                static_kernel = NPUStaticArtifactAdapter.from_compiled_kernel(
                    binary, compile_meta, inductor_meta
                )
            except (NPUStaticArtifactError, RuntimeError, TypeError, ValueError) as exc:
                raise CannotStaticallyLaunchNPUKernel(str(exc)) from exc

            metadata = static_kernel.npu_launch_metadata
            if metadata.workspace_size != 0:
                raise CannotStaticallyLaunchNPUKernel("workspace is required")
            if metadata.lock_num != 0:
                raise CannotStaticallyLaunchNPUKernel("sync block lock is required")
            if metadata.device_print_enabled:
                raise CannotStaticallyLaunchNPUKernel("device print is enabled")
            return static_kernel

        try:
            return check()
        except CannotStaticallyLaunchNPUKernel as exc:
            reason = str(exc)
            kernel_name = (
                inductor_meta.get("kernel_name")
                or getattr(binary, "name", None)
                or "unknown"
            )
            log.info(
                "Bypassing NPU static Triton launcher for kernel %s due to %s",
                kernel_name,
                reason,
            )
            if config.strict_static_triton_launcher:
                raise
            return None

    def reload_npubin_path(self) -> None:
        self.kernel.npu_launch_metadata.validate()
        directory = os.path.join(
            triton_cache_dir(self.compile_meta.get("device", 0) or 0),
            triton_hash_to_path_key(self.kernel.hash),
        )
        npubin_location = os.path.join(directory, f"{self.kernel.name}.npubin")
        if not os.path.exists(npubin_location):
            if self.kernel.npubin_raw is not None:
                self.kernel.reload_npubin_from_raw(npubin_location)
            else:
                raise RuntimeError(
                    "Npubin file saved by TritonBundler not found at "
                    f"{npubin_location}"
                )
        self.kernel.npubin_path = npubin_location

    def reload_cubin_path(self) -> None:
        # TritonBundler currently calls this CUDA-named compatibility method.
        self.reload_npubin_path()

    def make_launcher(self):
        set_feature_use("static_triton_launcher", True)
        if not self.kernel.npubin_path:
            self.reload_npubin_path()

        kernel = self.kernel
        kernel._init_handles()
        # The adapter owns constexpr/specialization filtering for the device ABI.
        # Keep outer arguments needed by grid code, but pass only runtime args.
        def_args = [
            *kernel.launcher_arg_names,
            *self.inductor_meta.get("extra_launcher_args", ()),
        ]
        runner_args = ["grid_0", "grid_1", "grid_2", "stream", *kernel.runtime_arg_names]
        scope = {"runner": kernel.run}
        grid_lines = _gen_grid_code(
            scope, def_args, kernel.arg_names, self.config, self.inductor_meta,
            num_cores=get_npu_vector_core_count(),
            is_a5=bool(self.inductor_meta.get("npu_dispatch_recipe")) and device_props.is_a5(),
        )
        # Use global bindings, matching the community clone-and-replace protocol.
        launcher = _gen_launcher_code(scope, def_args, runner_args, grid_lines)
        launcher.config = self.config
        launcher.runnable = True
        launcher.n_regs = kernel.n_regs
        launcher.n_spills = kernel.n_spills
        launcher.shared = kernel.shared
        launcher.cache_hash = (
            triton_hash_to_path_key(kernel.hash) if kernel.hash is not None else None
        )
        launcher.store_cubin = False
        launcher._is_static = True
        return launcher


__all__ = [
    "CannotStaticallyLaunchNPUKernel",
    "NPUStaticTritonCompileResult",
]
