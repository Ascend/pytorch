from __future__ import annotations

from importlib import import_module
from numbers import Real
from operator import index as operator_index
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .types import FastLaunchError, FastLaunchPlanUnavailable


_SUPPORTED_ARG_KINDS = frozenset(
    (
        "tensor",
        "i32",
        "i64",
        "u32",
        "u64",
        "f32",
        "f64",
        "bool",
    )
)
_INT32_MAX = 2**31 - 1
_UINT16_MAX = 2**16 - 1
_INTEGER_ARG_KINDS = frozenset(("i32", "i64", "u32", "u64"))
_FLOAT_ARG_KINDS = frozenset(("f32", "f64"))


def arg_kind_from_abi_signature(signature: Any) -> str | None:
    text = str(signature or "").strip().lower()
    if text.startswith(("*", "memref")) or "ptr" in text:
        return "tensor"
    if text in ("i1", "bool"):
        return "bool"
    if "u32" in text:
        return "u32"
    if "i32" in text or text == "int":
        return "i32"
    if "u64" in text:
        return "u64"
    if "i64" in text or text == "long":
        return "i64"
    if "fp32" in text or "f32" in text or text == "float":
        return "f32"
    if "fp64" in text or "f64" in text or text == "double":
        return "f64"
    return None


def _normalize_grid(grid: Any) -> tuple[int, int, int]:
    try:
        values = tuple(int(value) for value in grid)
    except (TypeError, ValueError) as exc:
        raise FastLaunchError(
            "invalid_grid",
            backend_submitted=False,
        ) from exc
    if len(values) != 3:
        raise FastLaunchError(
            f"grid_rank_mismatch:{len(values)}",
            backend_submitted=False,
        )
    product = 1
    for index, value in enumerate(values):
        if value <= 0:
            raise FastLaunchError(
                f"grid_dim_non_positive:{index}",
                backend_submitted=False,
            )
        if value > _INT32_MAX:
            raise FastLaunchError(
                f"grid_dim_exceeds_int32:{index}",
                backend_submitted=False,
            )
        product *= value
        if product > _UINT16_MAX:
            raise FastLaunchError(
                "grid_product_exceeds_uint16",
                backend_submitted=False,
            )
    return values


def _load_c_extension() -> Any:
    try:
        return import_module("torch_npu._C")
    except Exception as exc:
        raise FastLaunchPlanUnavailable("c_extension_unavailable") from exc


def _kernel_stub_supported(kernel_stub: Any) -> bool:
    if isinstance(kernel_stub, int):
        return kernel_stub != 0
    if type(kernel_stub).__name__ == "PyCapsule":
        return True
    if hasattr(kernel_stub, "value"):
        try:
            return int(kernel_stub.value) != 0
        except (TypeError, ValueError):
            return False
    try:
        return int(kernel_stub) != 0
    except (TypeError, ValueError):
        return False


def _callsite_schema_state(callsite_metadata: dict[str, Any]) -> str:
    state = callsite_metadata.get("schema_state")
    if state in ("complete", "incomplete", "conflict"):
        return str(state)

    # Generated wrappers from the first fast-launch revision only carried an
    # eligible bit. Missing/unknown codegen signatures can be completed from
    # the selected launcher ABI; an explicit non-codegen rejection remains a
    # conflict for backward compatibility.
    if callsite_metadata.get("eligible", False):
        return "complete" if callsite_metadata.get("arg_kinds") else "incomplete"
    fallback_reason = str(callsite_metadata.get("fallback_reason") or "")
    if fallback_reason.startswith("codegen_"):
        return "incomplete"
    return "conflict"


def _validate_callsite_schema(
    callsite_metadata: dict[str, Any],
    launcher_arg_kinds: tuple[str, ...],
    runtime_arg_count: int,
) -> None:
    state = _callsite_schema_state(callsite_metadata)
    if state == "conflict":
        raise FastLaunchPlanUnavailable(
            str(callsite_metadata.get("fallback_reason") or "codegen_schema_conflict")
        )
    if runtime_arg_count > len(launcher_arg_kinds):
        raise FastLaunchPlanUnavailable("launcher_schema_too_short")
    if state != "complete":
        return

    callsite_arg_kinds = tuple(callsite_metadata.get("arg_kinds", ()) or ())
    if len(callsite_arg_kinds) != runtime_arg_count:
        raise FastLaunchPlanUnavailable("codegen_schema_size_conflict")
    if callsite_arg_kinds != launcher_arg_kinds[:runtime_arg_count]:
        raise FastLaunchPlanUnavailable("codegen_launcher_schema_conflict")


def _runtime_arg_matches_kind(arg: Any, kind: str) -> bool:
    if kind == "tensor":
        return callable(getattr(arg, "data_ptr", None))
    if kind == "bool":
        return isinstance(arg, bool)
    if kind in _INTEGER_ARG_KINDS:
        if isinstance(arg, bool):
            return False
        try:
            operator_index(arg)
            return True
        except TypeError:
            return False
    if kind in _FLOAT_ARG_KINDS:
        return isinstance(arg, Real) and not isinstance(arg, bool)
    return False


def _validate_runtime_arg_categories(
    canonical_args: tuple[Any, ...],
    launcher_arg_kinds: tuple[str, ...],
) -> None:
    if len(canonical_args) != len(launcher_arg_kinds):
        raise FastLaunchPlanUnavailable(
            f"canonical_args_size_mismatch:{len(canonical_args)}:"
            f"{len(launcher_arg_kinds)}"
        )
    for arg_index, (arg, kind) in enumerate(zip(canonical_args, launcher_arg_kinds)):
        if not _runtime_arg_matches_kind(arg, kind):
            raise FastLaunchPlanUnavailable(
                f"runtime_arg_category_mismatch:{arg_index}:{kind}"
            )


class PlannedFastLaunch:
    __slots__ = (
        "arg_kinds",
        "get_grid",
        "launcher",
        "plan",
        "untimed_launch",
    )

    def __init__(
        self,
        *,
        launcher: Any,
        plan: Any,
        arg_kinds: tuple[str, ...],
        get_grid: Callable[..., Any],
        untimed_launch: Callable[..., Any],
    ) -> None:
        self.launcher = launcher
        self.plan = plan
        self.arg_kinds = arg_kinds
        self.get_grid = get_grid
        self.untimed_launch = untimed_launch

    def __call__(
        self,
        args: tuple[Any, ...],
        *,
        stream: Any,
    ) -> None:
        if len(args) != len(self.arg_kinds):
            raise FastLaunchError(
                f"args_size_mismatch:{len(args)}:{len(self.arg_kinds)}",
                backend_submitted=False,
                stable=True,
            )
        if stream is None:
            raise FastLaunchError(
                "stream_is_none",
                backend_submitted=False,
            )
        try:
            grid = _normalize_grid(self.get_grid(*args))
        except FastLaunchError:
            raise
        except Exception as exc:
            raise FastLaunchError(
                f"grid_resolve_error:{type(exc).__name__}",
                backend_submitted=False,
            ) from exc

        try:
            self.untimed_launch(
                self.plan,
                stream,
                grid[0],
                grid[1],
                grid[2],
                args,
            )
        except Exception as exc:
            # All recoverable validation is completed before entering C++.
            # Treat errors after the boundary as submitted so fallback can never
            # replay an already queued kernel.
            raise FastLaunchError(
                f"backend_error:{type(exc).__name__}:{exc}",
                backend_submitted=True,
            ) from exc
        return None


def build_planned_fast_launch(
    launcher: Any,
    callsite_metadata: dict[str, Any],
    *,
    canonical_args: tuple[Any, ...] | None = None,
    runtime_arg_count: int | None = None,
) -> PlannedFastLaunch:
    kernel_name = str(getattr(launcher, "_npu_fast_launch_kernel_name", "") or "")
    kernel_stub = getattr(launcher, "_npu_fast_launch_kernel_stub", None)
    kernel_stub_owner = getattr(launcher, "_npu_fast_launch_kernel_stub_owner", None)
    get_grid = getattr(launcher, "_npu_fast_launch_get_grid", None)
    arg_kinds = tuple(getattr(launcher, "_npu_fast_launch_arg_kinds", ()) or ())
    if not kernel_name:
        raise FastLaunchPlanUnavailable("kernel_name_missing")
    if not _kernel_stub_supported(kernel_stub):
        raise FastLaunchPlanUnavailable("kernel_stub_unsupported")
    if kernel_stub_owner is None:
        raise FastLaunchPlanUnavailable("kernel_stub_owner_missing")
    if not callable(get_grid):
        raise FastLaunchPlanUnavailable("grid_resolver_missing")
    # Triton keeps launch hook objects in the generated launcher scope even
    # for launchers that the established planned fast path can invoke
    # directly. Do not make their mere presence a permanent negative cache.
    # BoundFastLaunch checks active hook callbacks on every direct call.
    if int(getattr(launcher, "_npu_fast_launch_workspace_size", 0) or 0) > 0:
        raise FastLaunchPlanUnavailable("launcher_workspace_required")
    if int(getattr(launcher, "_npu_fast_launch_lock_num", 0) or 0) > 0:
        raise FastLaunchPlanUnavailable("launcher_sync_block_lock_required")
    if getattr(launcher, "_npu_fast_launch_device_print_enabled", False):
        raise FastLaunchPlanUnavailable("launcher_device_print_required")
    target_support_ffts = getattr(
        launcher, "_npu_fast_launch_target_support_ffts", None
    )
    if target_support_ffts is None:
        raise FastLaunchPlanUnavailable("launcher_ffts_abi_unknown")
    if not arg_kinds or any(kind not in _SUPPORTED_ARG_KINDS for kind in arg_kinds):
        raise FastLaunchPlanUnavailable("arg_kinds_unsupported")

    if runtime_arg_count is None:
        runtime_arg_count = int(
            callsite_metadata.get("runtime_arg_count", len(arg_kinds))
        )
    _validate_callsite_schema(callsite_metadata, arg_kinds, runtime_arg_count)
    if canonical_args is not None:
        _validate_runtime_arg_categories(canonical_args, arg_kinds)

    extension = _load_c_extension()
    make_plan = getattr(extension, "_npu_inductor_make_fast_launch_plan", None)
    launch = getattr(extension, "_npu_inductor_fast_launch_with_plan", None)
    if not callable(make_plan) or not callable(launch):
        raise FastLaunchPlanUnavailable("planned_backend_unavailable")

    enable_simt = bool(getattr(launcher, "_npu_fast_launch_enable_simt", False))
    shared_mem_dynamic_size = int(
        getattr(launcher, "_npu_fast_launch_shared_mem_dynamic_size", 0) or 0
    )
    is_pure_simt = bool(getattr(launcher, "_npu_fast_launch_force_simt_only", False))
    try:
        plan = make_plan(
            kernel_name,
            kernel_stub,
            arg_kinds,
            enable_simt,
            shared_mem_dynamic_size,
            is_pure_simt,
            bool(target_support_ffts),
        )
        # The C++ plan owns the stub object; this additional reference owns the
        # loaded binary that produced it.
        plan._owner = kernel_stub_owner
    except Exception as exc:
        raise FastLaunchPlanUnavailable(
            f"plan_creation_error:{type(exc).__name__}"
        ) from exc

    return PlannedFastLaunch(
        launcher=launcher,
        plan=plan,
        arg_kinds=arg_kinds,
        get_grid=get_grid,
        untimed_launch=launch,
    )


__all__ = [
    "PlannedFastLaunch",
    "arg_kind_from_abi_signature",
    "build_planned_fast_launch",
]
