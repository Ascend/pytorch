from __future__ import annotations

import os
from typing import Any

from .backend import arg_kind_from_abi_signature


def _metadata_value(metadata: Any, name: str, default: Any) -> Any:
    if isinstance(metadata, dict):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _launch_hook_callbacks(hook: Any) -> Any:
    if hook is None:
        return ()

    calls = getattr(hook, "calls", None)
    if (
        type(hook).__name__ == "HookChain"
        and calls is not None
        and callable(getattr(hook, "add", None))
        and callable(getattr(hook, "remove", None))
    ):
        # Triton keeps one HookChain object in generated launcher scopes and
        # mutates its calls list when instrumentation starts or stops. Keep the
        # list itself so the steady-state check is both dynamic and cheap.
        return calls

    # Older Triton versions expose a single hook callable instead of a
    # HookChain. Its presence means that the original launcher is required.
    return (hook,)


def attach_python_wrapper_launcher_metadata(
    launcher: Any,
    *,
    kernel_name: str,
    kernel_stub: Any,
    kernel_stub_owner: Any,
    get_grid: Any,
    grid: Any,
    def_args: Any,
    compile_meta: dict[str, Any],
    binary: Any,
    launcher_enter: Any,
    launcher_exit: Any,
) -> Any:
    signature = compile_meta.get("signature", {}) or {}
    arg_signatures = tuple(str(signature.get(name, "")) for name in def_args)
    launcher._npu_fast_launch_kernel_name = str(kernel_name)
    launcher._npu_fast_launch_kernel_stub = kernel_stub
    launcher._npu_fast_launch_kernel_stub_owner = kernel_stub_owner
    launcher._npu_fast_launch_get_grid = get_grid
    launcher._npu_fast_launch_grid_exprs = (
        str(grid.x_grid),
        str(grid.y_grid),
        str(grid.z_grid),
    )
    launcher._npu_fast_launch_def_args = tuple(str(name) for name in def_args)
    launcher._npu_fast_launch_arg_signatures = arg_signatures
    launcher._npu_fast_launch_arg_kinds = tuple(
        arg_kind_from_abi_signature(value) for value in arg_signatures
    )

    metadata = getattr(binary, "metadata", None)
    parallel_mode = _metadata_value(metadata, "parallel_mode", "")
    is_pure_simt = _metadata_value(metadata, "is_pure_simt", False)
    shared_mem_dynamic_size = _metadata_value(
        metadata,
        "shared_mem_dynamic_size",
        0,
    )
    workspace_size = _metadata_value(metadata, "workspace_size", -1)
    lock_num = _metadata_value(metadata, "lock_num", -1)

    launcher._npu_fast_launch_enable_simt = "simt" in str(
        parallel_mode
    ).lower() or bool(is_pure_simt)
    launcher._npu_fast_launch_is_pure_simt = bool(is_pure_simt)
    launcher._npu_fast_launch_shared_mem_dynamic_size = int(
        shared_mem_dynamic_size or 0
    )

    # The Ascend runner prepends the FFTS synchronization address to the
    # packed ABI when the target supports FFTS. Preserve that compile-time
    # decision so the planned backend builds the same hidden-argument layout.
    from torch_npu._inductor.utils import triton_support_ffts

    launcher._npu_fast_launch_target_support_ffts = bool(triton_support_ffts())
    launcher._npu_fast_launch_workspace_size = int(workspace_size or 0)
    launcher._npu_fast_launch_lock_num = int(lock_num or 0)
    launcher._npu_fast_launch_device_print_enabled = os.getenv(
        "TRITON_DEVICE_PRINT",
        "false",
    ).lower() in ("true", "1")
    launcher._npu_fast_launch_enter_hook = launcher_enter
    launcher._npu_fast_launch_exit_hook = launcher_exit
    launcher._npu_fast_launch_enter_hook_callbacks = _launch_hook_callbacks(
        launcher_enter
    )
    launcher._npu_fast_launch_exit_hook_callbacks = _launch_hook_callbacks(
        launcher_exit
    )
    return launcher


__all__ = ["attach_python_wrapper_launcher_metadata"]
