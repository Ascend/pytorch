# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

# Mirrored from: pytorch/torch/_inductor/runtime/static_triton_launcher.py
# Upstream commit: b62a09d1dd4d337a7d41b73a5c2e38744f9d5ddc
# Local differences: NPU binary format, metadata, CANN execution and ownership.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metadata import NPULaunchMetadata


@dataclass
class _StaticFunction:
    arg_names: tuple[str, ...]
    constexprs: tuple[int, ...]


@dataclass
class _StaticSource:
    fn: _StaticFunction


class NPUStaticallyLaunchedTritonKernel:
    launch_enter_hook = None
    launch_exit_hook = None

    def __init__(
        self,
        *,
        name: str,
        npubin_raw: bytes | None,
        npubin_path: str | None,
        kernel_hash: str,
        arg_names: tuple[str, ...],
        launcher_arg_names: tuple[str, ...],
        runtime_arg_names: tuple[str, ...],
        declared_constexprs: tuple[int, ...],
        full_constexprs: tuple[int, ...],
        arg_kinds: tuple[str, ...],
        launch_metadata: NPULaunchMetadata,
        device: int,
        num_warps: int,
        shared: int,
        n_regs: int | None,
        n_spills: int | None,
    ) -> None:
        if not name:
            raise RuntimeError("NPU static launcher kernel name is missing")
        if not kernel_hash:
            raise RuntimeError("NPU static launcher kernel hash is missing")
        self.name = name
        self.npubin_raw = npubin_raw
        self.npubin_path = npubin_path
        self.hash = kernel_hash
        self.arg_names = arg_names
        self.launcher_arg_names = launcher_arg_names
        self.runtime_arg_names = runtime_arg_names
        launcher_arg_positions = {
            name: index for index, name in enumerate(launcher_arg_names)
        }
        try:
            self.runtime_arg_indices = tuple(
                launcher_arg_positions[name] for name in runtime_arg_names
            )
        except KeyError as exc:
            raise RuntimeError(
                f"NPU static launcher runtime argument {exc.args[0]!r} is missing"
            ) from exc
        self.declared_constexprs = declared_constexprs
        self.full_constexprs = full_constexprs
        self.arg_kinds = arg_kinds
        self.npu_launch_metadata = launch_metadata
        # The existing NPU launcher generator expects binary.metadata, while
        # binary.launch_metadata is reserved for Triton's callable hook API.
        self.metadata = launch_metadata
        self.device = device
        self.num_warps = num_warps
        self.shared = shared
        self.n_regs = 0 if n_regs is None else n_regs
        self.n_spills = 0 if n_spills is None else n_spills
        self.loaded_kernel: Any | None = None
        self.function: Any | None = None
        self._c_impl: Any | None = None
        self.src = _StaticSource(_StaticFunction(arg_names, declared_constexprs))

    def _C_impl(self):
        if self._c_impl is None:
            from torch_npu import _C

            self._c_impl = getattr(_C, "_StaticNpuLauncher", None)
            if self._c_impl is None:
                raise RuntimeError("torch_npu was built without the NPU static launcher")
        return self._c_impl

    def reload_npubin_from_raw(self, filepath: str) -> str:
        if self.npubin_raw is None:
            raise RuntimeError("npubin_raw is unavailable")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.npubin_raw)
        self.npubin_path = str(path)
        return self.npubin_path

    def load_kernel(self, device: int | None = None) -> None:
        if self.loaded_kernel is not None:
            return
        self.npu_launch_metadata.validate()
        if self.npubin_raw is None:
            if not self.npubin_path:
                raise RuntimeError("NPU static launcher has no npubin source")
            self.npubin_raw = Path(self.npubin_path).read_bytes()
        target_device = self.device if device is None else int(device)
        metadata = self.npu_launch_metadata
        loaded_kernel = self._C_impl()._load_kernel(
            self.npubin_raw,
            self.name,
            target_device,
            self.arg_kinds,
            metadata.mix_mode,
            metadata.enable_simt,
            metadata.shared_mem_dynamic_size,
            metadata.is_pure_simt,
            metadata.target_support_ffts,
            metadata.trailing_pointer_count,
        )
        self.loaded_kernel = loaded_kernel
        # Expose an opaque owning object, not a second raw handle lifetime.
        self.function = loaded_kernel
        # The bundle copy is made before launchers are materialized. Once this
        # runtime instance is loaded, release its duplicate binary bytes/path.
        self.npubin_raw = None
        self.npubin_path = None

    def _init_handles(self) -> None:
        self.load_kernel(self.device)

    def close(self) -> None:
        loaded_kernel = self.loaded_kernel
        if loaded_kernel is None:
            return
        self.loaded_kernel = None
        self.function = None
        self._C_impl()._unload_kernel(loaded_kernel)

    def __del__(self) -> None:
        # Dropping loaded_kernel transfers final cleanup to its C++ owner.
        self.loaded_kernel = None
        self.function = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["loaded_kernel"] = None
        state["function"] = None
        state["npubin_path"] = None
        state["_c_impl"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._c_impl = None
        if not hasattr(self, "runtime_arg_indices"):
            launcher_arg_positions = {
                name: index for index, name in enumerate(self.launcher_arg_names)
            }
            self.runtime_arg_indices = tuple(
                launcher_arg_positions[name] for name in self.runtime_arg_names
            )

    def run(
        self,
        grid_0: int,
        grid_1: int,
        grid_2: int,
        stream: int,
        *args: object,
    ) -> None:
        if self.loaded_kernel is None:
            raise RuntimeError("load_kernel() must be called before run()")
        if len(args) != len(self.runtime_arg_names):
            raise RuntimeError(
                "NPU static launcher received an unexpected argument count: "
                f"{len(args)} vs {len(self.runtime_arg_names)}"
            )
        self._C_impl()._launch_kernel(
            self.loaded_kernel,
            int(grid_0),
            int(grid_1),
            int(grid_2),
            int(stream),
            args,
        )


__all__ = ["NPUStaticallyLaunchedTritonKernel"]
