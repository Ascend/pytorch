# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Lowering snapshot / restore for multi-backend (Triton vs MLIR/DVM) switching.
#
# Kept in a dedicated module so ``__init__.py`` only orchestrates *when* to
# patch; the mechanics of capture/restore/apply live here and can be tested
# independently.

from __future__ import annotations
import copy
import functools
import importlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.virtualized import V

from .lowering_common import LOWERING_REGISTRY_ATTRS, get_module_functions

_BASELINE: Optional["LoweringSnapshot"] = None
_INDUCTOR_ATTR_BASELINE = None
_DEVICE_DISPATCH_MARKER = "_torch_npu_device_lowering_dispatch"
_UPSTREAM_HANDLER_ATTR = "_torch_npu_upstream_handler"
_DEVICE_HANDLER_ATTR = "_torch_npu_device_handler"

_DeviceLoweringPredicate = Callable[[tuple[Any, ...], dict[str, Any], str], bool]


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


def _expand_lowering_targets(ops: Iterable[Any]) -> list[Any]:
    targets = []
    seen = set()
    for op in ops:
        candidates = [op]
        if isinstance(op, torch._ops.OpOverloadPacket):
            candidates.extend(op.op_overloads())
        for target in candidates:
            if target not in seen:
                seen.add(target)
                targets.append(target)
    return targets


def _iter_ir_device_types(value: Any):
    for leaf in pytree.tree_leaves(value):
        get_device = getattr(leaf, "get_device", None)
        if not callable(get_device):
            continue
        try:
            device = get_device()
        except NotImplementedError:
            continue
        device_type = getattr(device, "type", None)
        if device_type is not None:
            yield device_type


def _cpu_scalar_fp64_cast_needs_device_lowering(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device_type: str,
) -> bool:
    """Keep CPU fp64 scalar casts out of generated NPU kernels."""
    if (
        device_type != "npu"
        or not args
        or device_type not in getattr(V.graph, "device_types", ())
    ):
        return False

    destination_dtype = args[1] if len(args) > 1 else kwargs.get("dtype")
    source = args[0]
    return (
        destination_dtype is not None
        and destination_dtype != torch.float64
        and source.get_device().type == "cpu"
        and source.get_dtype() == torch.float64
        and len(source.get_size()) == 0
    )


_DEFAULT_EXTRA_DEVICE_PREDICATES = {
    torch.ops.prims.convert_element_type: (
        _cpu_scalar_fp64_cast_needs_device_lowering
    ),
}


def _uses_device_lowering(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device_type: str,
    extra_device_predicate: Optional[_DeviceLoweringPredicate] = None,
) -> bool:
    layout = kwargs.get("layout")
    layout_device_type = getattr(
        getattr(layout, "device", None), "type", None
    )
    if layout_device_type is not None:
        return layout_device_type == device_type
    if os.environ.get("INDUCTOR_ASCEND_DUMP_FX_GRAPH") or os.environ.get("INDUCTOR_ASCEND_CHECK_ACCURACY"):
        device_kwarg = kwargs.get("device")
        if isinstance(device_kwarg, (torch.device, str)):
            try:
                kwarg_device_type = torch.device(device_kwarg).type
            except Exception:
                kwarg_device_type = None
            if kwarg_device_type is not None:
                return kwarg_device_type in ("npu", "cpu")
    if device_type in _iter_ir_device_types((args, kwargs)):
        return True
    return extra_device_predicate is not None and extra_device_predicate(
        args, kwargs, device_type
    )


def _make_device_lowering_dispatcher(
    upstream_handler: Callable[..., Any],
    device_handler: Callable[..., Any],
    device_type: str,
    extra_device_predicate: Optional[_DeviceLoweringPredicate] = None,
) -> Callable[..., Any]:
    @functools.wraps(device_handler)
    def dispatcher(*args, **kwargs):
        handler = (
            device_handler
            if _uses_device_lowering(
                args, kwargs, device_type, extra_device_predicate
            )
            else upstream_handler
        )
        return handler(*args, **kwargs)

    setattr(dispatcher, _DEVICE_DISPATCH_MARKER, True)
    setattr(dispatcher, _UPSTREAM_HANDLER_ATTR, upstream_handler)
    setattr(dispatcher, _DEVICE_HANDLER_ATTR, device_handler)
    return dispatcher


def _expand_extra_device_predicates(
    predicates: Optional[Mapping[Any, _DeviceLoweringPredicate]],
) -> dict[Any, _DeviceLoweringPredicate]:
    if not predicates:
        return {}

    expanded = dict(predicates)
    for op, predicate in predicates.items():
        for target in _expand_lowering_targets((op,)):
            expanded.setdefault(target, predicate)
    return expanded


def install_device_lowering_dispatch(
    ops: Iterable[Any],
    device_type: str = "npu",
    extra_device_predicates: Optional[Mapping[Any, _DeviceLoweringPredicate]] = None,
) -> None:
    baseline = capture_lowering_baseline()
    lowering = _get_inductor_lowering()
    predicates = dict(_DEFAULT_EXTRA_DEVICE_PREDICATES)
    if extra_device_predicates:
        predicates.update(extra_device_predicates)
    expanded_predicates = _expand_extra_device_predicates(predicates)

    for target in _expand_lowering_targets(ops):
        upstream_handler = baseline.lowerings_copy.get(target)
        device_handler = lowering.lowerings.get(target)
        if (
            upstream_handler is None
            or device_handler is None
            or device_handler is upstream_handler
            or getattr(device_handler, _DEVICE_DISPATCH_MARKER, False)
        ):
            continue
        lowering.lowerings[target] = _make_device_lowering_dispatcher(
            upstream_handler,
            device_handler,
            device_type,
            expanded_predicates.get(target),
        )


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


def _snapshot_inductor_attr(owner, name):
    return owner, name, hasattr(owner, name), getattr(owner, name, None)


def _get_inductor_attr_baseline():
    global _INDUCTOR_ATTR_BASELINE
    if _INDUCTOR_ATTR_BASELINE is not None:
        return _INDUCTOR_ATTR_BASELINE

    from torch._inductor import scheduler as inductor_scheduler

    Scheduler = inductor_scheduler.Scheduler
    _INDUCTOR_ATTR_BASELINE = (
        _snapshot_inductor_attr(Scheduler, "_codegen"),
        _snapshot_inductor_attr(Scheduler, "compute_ancestors"),
        _snapshot_inductor_attr(inductor_scheduler, "_prune_redundant_deps"),
        _snapshot_inductor_attr(Scheduler, "can_fuse_vertical"),
        _snapshot_inductor_attr(Scheduler, "_get_unmet_dep_nodes"),
    )
    return _INDUCTOR_ATTR_BASELINE


def restore_inductor_baseline() -> None:
    """Reset lowering and scheduler hooks before loading a new NPU backend."""
    attr_baseline = _get_inductor_attr_baseline()
    restore_lowering_baseline()
    for owner, name, exists, value in attr_baseline:
        if exists:
            setattr(owner, name, value)
        elif hasattr(owner, name):
            delattr(owner, name)

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
