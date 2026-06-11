# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Lowering snapshot / restore for multi-backend (Triton vs MLIR/DVM) switching.
#
# Kept in a dedicated module so ``__init__.py`` only orchestrates *when* to
# patch; the mechanics of capture/restore/apply live here and can be tested
# independently.

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .lowering_common import LOWERING_REGISTRY_ATTRS, get_module_functions

_BASELINE: Optional["LoweringSnapshot"] = None


@dataclass
class LoweringSnapshot:
    """Pristine torch._inductor.lowering state captured before any NPU patch."""

    functions: dict[str, Callable[..., Any]]
    lowerings_ref: dict[Any, Any]
    lowerings_copy: dict[Any, Any]
    registry_copies: dict[str, Any] = field(default_factory=dict)
    make_reduction: Any = None


def _get_inductor_lowering():
    from torch._inductor import lowering as inductor_lowering

    return inductor_lowering


def _copy_registry_value(value: Any) -> Any:
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:
            pass
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (set, list)):
        return type(value)(value)
    return copy.copy(value)


def _module_functions(module: Any) -> dict[str, Callable[..., Any]]:
    return get_module_functions(module)


def capture_lowering_baseline() -> LoweringSnapshot:
    """Capture PT lowering once; safe to call repeatedly."""
    global _BASELINE
    if _BASELINE is not None:
        return _BASELINE

    lowering = _get_inductor_lowering()
    registry_copies = {
        attr: _copy_registry_value(getattr(lowering, attr))
        for attr in LOWERING_REGISTRY_ATTRS
    }
    _BASELINE = LoweringSnapshot(
        functions=_module_functions(lowering),
        lowerings_ref=lowering.lowerings,
        lowerings_copy=dict(lowering.lowerings),
        registry_copies=registry_copies,
        make_reduction=getattr(lowering, "make_reduction", None),
    )
    return _BASELINE


def restore_lowering_baseline() -> None:
    """Reset torch._inductor.lowering to the captured PT baseline."""
    baseline = capture_lowering_baseline()
    lowering = _get_inductor_lowering()

    for name, func in baseline.functions.items():
        if hasattr(lowering, name):
            setattr(lowering, name, func)

    if lowering.lowerings is not baseline.lowerings_ref:
        lowering.lowerings = baseline.lowerings_ref
    baseline.lowerings_ref.clear()
    baseline.lowerings_ref.update(baseline.lowerings_copy)

    for attr in LOWERING_REGISTRY_ATTRS:
        target = getattr(lowering, attr)
        snapshot_value = baseline.registry_copies[attr]
        if hasattr(target, "clear") and hasattr(target, "update"):
            target.clear()
            target.update(snapshot_value)
        elif isinstance(target, dict):
            target.clear()
            target.update(snapshot_value)
        else:
            setattr(lowering, attr, _copy_registry_value(snapshot_value))

    lowering.make_reduction = baseline.make_reduction


def merge_missing_lowerings(
    target_lowerings: dict[Any, Any],
    source_lowerings: dict[Any, Any],
) -> None:
    extra_keys = set(source_lowerings.keys()) - set(target_lowerings.keys())
    if extra_keys:
        target_lowerings.update({k: source_lowerings[k] for k in extra_keys})


def apply_mlir_lowering_patch(npu_lowering_module: Any) -> None:
    """Replace torch._inductor.lowering with the MLIR/DVM fork."""
    from torch._inductor import graph, lowering as inductor_lowering

    npu_functions = _module_functions(npu_lowering_module)
    inductor_functions = _module_functions(inductor_lowering)
    for name in inductor_functions:
        if name in npu_functions:
            setattr(inductor_lowering, name, npu_functions[name])

    merge_missing_lowerings(
        npu_lowering_module.lowerings,
        inductor_lowering.lowerings,
    )

    for attr in LOWERING_REGISTRY_ATTRS:
        setattr(inductor_lowering, attr, getattr(npu_lowering_module, attr))

    importlib.reload(graph)


def apply_mlir_inductor_patch() -> None:
    """MLIR/DVM: patch lowering + scheduler (called from _load_backend)."""
    from .ascend_npu_ir.ascend_npu_ir.npu.inductor_patch import lowering as npu_lowering
    from .ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.scheduler import (
        _patch_scheduler,
    )

    # Ensure IR patches (TracedGraph hooks) are registered.
    import torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.inductor_patch.ir  # noqa: F401

    apply_mlir_lowering_patch(npu_lowering)
    _patch_scheduler()
