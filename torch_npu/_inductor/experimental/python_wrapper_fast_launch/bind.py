from __future__ import annotations

from typing import Any

import torch.autograd.profiler as autograd_profiler

from torch_npu._inductor import config as npu_config

from .backend import PlannedFastLaunch, build_planned_fast_launch
from .types import FastLaunchError, FastLaunchPlanUnavailable


_MISSING_HOOK_CALLBACKS = object()


def _static_full_entry_reason(autotuner: Any) -> str | None:
    if not npu_config.enable_fast_launch:
        return "disabled"
    if getattr(npu_config, "dump_fx_graph", False):
        return "dump_fx_graph"
    if getattr(npu_config, "check_accuracy", False):
        return "check_accuracy"
    if getattr(autotuner, "triton_interpret", False):
        return "triton_interpret"
    if getattr(autotuner, "dump_launch_params", False):
        return "dump_launch_params"
    return None


def _is_grouped_autotuner(autotuner: Any) -> bool:
    inductor_meta = getattr(autotuner, "inductor_meta", {}) or {}
    return bool(
        inductor_meta.get("group_enabled", False)
        or hasattr(autotuner, "best_launcher_map")
    )


def _store_cubin_pending(autotuner: Any, launcher: Any) -> bool:
    return bool(
        getattr(launcher, "store_cubin", False)
        and not getattr(autotuner, "cuda_kernel_saved", False)
    )


def _launcher_has_active_launch_hooks(launcher: Any) -> bool:
    enter_callbacks = getattr(
        launcher,
        "_npu_fast_launch_enter_hook_callbacks",
        _MISSING_HOOK_CALLBACKS,
    )
    exit_callbacks = getattr(
        launcher,
        "_npu_fast_launch_exit_hook_callbacks",
        _MISSING_HOOK_CALLBACKS,
    )
    if (
        enter_callbacks is _MISSING_HOOK_CALLBACKS
        and exit_callbacks is _MISSING_HOOK_CALLBACKS
    ):
        # Conservatively preserve behavior for launchers carrying metadata
        # produced by the earlier boolean-only implementation.
        return bool(getattr(launcher, "_npu_fast_launch_has_launch_hooks", False))
    if (
        enter_callbacks is not _MISSING_HOOK_CALLBACKS
        and enter_callbacks
    ):
        return True
    return bool(
        exit_callbacks is not _MISSING_HOOK_CALLBACKS
        and exit_callbacks
    )


class BoundFastLaunch:
    __slots__ = (
        "autotuner",
        "metadata",
        "_direct",
        "_call_slot",
        "_negative_callable",
        "_negative_launcher",
        "_static_full_entry_reason",
    )

    def __init__(
        self,
        autotuner: Any,
        metadata: dict[str, Any],
        *,
        call_slot: list[Any] | None = None,
    ) -> None:
        self.autotuner = autotuner
        self.metadata = dict(metadata)
        self._direct: PlannedFastLaunch | None = None
        self._call_slot = call_slot
        self._static_full_entry_reason = _static_full_entry_reason(autotuner)
        self._negative_launcher: Any | None = None
        self._negative_callable: Any | None = None

    def _stable_launcher(self) -> Any | None:
        launchers = tuple(getattr(self.autotuner, "launchers", ()) or ())
        if len(launchers) != 1:
            return None
        launcher = launchers[0]
        if hasattr(launcher, "fallback"):
            return None
        best_launcher = getattr(self.autotuner, "best_launcher", None)
        if best_launcher is None or best_launcher is not launcher:
            return None
        if _store_cubin_pending(self.autotuner, launcher):
            return None
        return launcher

    def _canonical_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        runtime_blocks = tuple(getattr(self.autotuner, "best_runtime_blocks", ()) or ())
        if not runtime_blocks:
            return args
        builder = getattr(self.autotuner, "_build_runtime_launch_args", None)
        if callable(builder):
            return tuple(builder(args, runtime_blocks))
        raise FastLaunchError(
            "runtime_block_builder_missing",
            backend_submitted=False,
            stable=True,
        )

    def _clear_negative(self) -> None:
        self._negative_launcher = None
        self._negative_callable = None

    def _install_negative(self, launcher: Any) -> None:
        self._direct = None
        self._negative_launcher = launcher
        self._negative_callable = launcher
        if self._call_slot is not None:
            self._call_slot[0] = self

    def _try_promote(self, args: tuple[Any, ...]) -> bool:
        launcher = self._stable_launcher()
        if launcher is None:
            return False
        if self._negative_launcher is launcher:
            return False
        if self._direct is not None and self._direct.launcher is launcher:
            return True
        try:
            canonical_args = self._canonical_args(args)
            self._direct = build_planned_fast_launch(
                launcher,
                self.metadata,
                canonical_args=canonical_args,
                runtime_arg_count=len(args),
            )
        except FastLaunchPlanUnavailable as exc:
            if exc.stable:
                self._install_negative(launcher)
            return False
        except FastLaunchError as exc:
            if exc.stable:
                self._install_negative(launcher)
            return False
        self._clear_negative()
        if self._call_slot is not None:
            self._call_slot[0] = FinalizedFastLaunch(self, self._direct)
        return True

    def _fallback(
        self,
        args: tuple[Any, ...],
        *,
        stream: Any,
        benchmark_run: bool,
        kwargs: dict[str, Any],
    ) -> Any:
        return self.autotuner.run(
            *args,
            stream=stream,
            benchmark_run=benchmark_run,
            **kwargs,
        )

    def _call_launcher(
        self,
        launcher_call: Any,
        args: tuple[Any, ...],
        *,
        stream: Any,
    ) -> Any:
        canonical_args = self._canonical_args(args)
        return launcher_call(*canonical_args, stream=stream)

    def _negative_fallback(
        self,
        launcher_call: Any,
        args: tuple[Any, ...],
        *,
        stream: Any,
    ) -> Any:
        return self._call_launcher(launcher_call, args, stream=stream)

    def _hook_fallback(
        self,
        launcher_call: Any,
        args: tuple[Any, ...],
        *,
        stream: Any,
    ) -> Any:
        return self._call_launcher(launcher_call, args, stream=stream)

    def _call_direct(
        self,
        direct: PlannedFastLaunch,
        args: tuple[Any, ...],
        *,
        stream: Any,
    ) -> Any:
        if _launcher_has_active_launch_hooks(direct.launcher):
            # Hook registration is dynamic. Use the selected original launcher
            # for this call only, keep the plan, and resume direct launch as
            # soon as the HookChain becomes empty again.
            return self._hook_fallback(
                direct.launcher,
                args,
                stream=stream,
            )
        try:
            direct(args, stream=stream)
        except FastLaunchError as exc:
            if exc.backend_submitted:
                raise
            if exc.stable:
                self._install_negative(direct.launcher)
            return self._fallback(
                args,
                stream=stream,
                benchmark_run=False,
                kwargs={},
            )
        return None

    def __call__(
        self,
        *args: Any,
        stream: Any,
        benchmark_run: bool = False,
        **kwargs: Any,
    ) -> Any:
        reason = self._static_full_entry_reason
        if reason is None and benchmark_run:
            reason = "benchmark_run"
        if reason is None and kwargs:
            reason = "runtime_kwargs"
        if reason is None and autograd_profiler._is_profiler_enabled:
            reason = "profiler"
        if reason is not None:
            return self._fallback(
                args,
                stream=stream,
                benchmark_run=benchmark_run,
                kwargs=kwargs,
            )

        direct = self._direct
        if direct is not None:
            # The plan is created only after all launcher lifecycle work is
            # complete. The selected launcher identity is the only mutable
            # state that must be rechecked on a steady-state hit.
            if getattr(self.autotuner, "best_launcher", None) is direct.launcher:
                return self._call_direct(direct, args, stream=stream)
            self._direct = None

        negative_launcher = self._negative_launcher
        if negative_launcher is not None:
            if getattr(self.autotuner, "best_launcher", None) is negative_launcher:
                return self._negative_fallback(
                    self._negative_callable,
                    args,
                    stream=stream,
                )
            self._clear_negative()

        if self._try_promote(args) and self._direct is not None:
            return self._call_direct(self._direct, args, stream=stream)

        result = self._fallback(
            args,
            stream=stream,
            benchmark_run=False,
            kwargs={},
        )
        # Promotion is deliberately after the original call: autotune,
        # coordinate descent, store-cubin, and debug semantics must complete
        # before the planned path may observe a launcher as stable.
        self._try_promote(args)
        return result


class FinalizedFastLaunch:
    """Steady-state entry installed directly into a generated call slot."""

    __slots__ = ("bound", "direct")

    def __init__(self, bound: BoundFastLaunch, direct: PlannedFastLaunch) -> None:
        self.bound = bound
        self.direct = direct

    def __call__(
        self,
        *args: Any,
        stream: Any,
        benchmark_run: bool = False,
        **kwargs: Any,
    ) -> Any:
        if (
            benchmark_run
            or kwargs
            or autograd_profiler._is_profiler_enabled
            or getattr(self.bound.autotuner, "best_launcher", None)
            is not self.direct.launcher
        ):
            return self.bound(
                *args,
                stream=stream,
                benchmark_run=benchmark_run,
                **kwargs,
            )
        if _launcher_has_active_launch_hooks(self.direct.launcher):
            return self.bound._hook_fallback(
                self.direct.launcher,
                args,
                stream=stream,
            )
        try:
            self.direct(args, stream=stream)
        except FastLaunchError as exc:
            if exc.backend_submitted:
                raise
            if exc.stable:
                self.bound._install_negative(self.direct.launcher)
            return self.bound._fallback(
                args,
                stream=stream,
                benchmark_run=False,
                kwargs={},
            )
        return None


def bind_python_wrapper_kernel_fast(
    metadata: dict[str, Any],
    autotuner: Any,
    *,
    call_slot: list[Any] | None = None,
) -> Any:
    bound = (
        autotuner.run
        if _is_grouped_autotuner(autotuner)
        else BoundFastLaunch(autotuner, metadata, call_slot=call_slot)
    )
    if call_slot is not None:
        call_slot[0] = bound
    return bound


__all__ = [
    "BoundFastLaunch",
    "FinalizedFastLaunch",
    "bind_python_wrapper_kernel_fast",
]
