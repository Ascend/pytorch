# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Ownership routing for graph-owned compiler cache artifacts.

These helpers decide *whose* record a restored object is; they do not inspect a
backend's configuration. That keeps them safe to import from the package
adapters, which run for every NPU compile.
"""
import logging

from .format import PackageCompatibilityError

log = logging.getLogger(__name__)


def bypass(reason):
    """Count and log one package/cache bypass."""
    from torch._dynamo.utils import counters

    counters["npu_package"][reason.split(":", 1)[0]] += 1
    log.info("NPU package bypass: %s", reason)


def walk_objects(value):
    """Traverse cache data without executing arbitrary properties or tensor ops."""
    import torch

    visited, pending = set(), [value]
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        yield item
        if isinstance(item, torch.Tensor):
            continue
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            # Compiler records are dataclasses/simple records. Do not descend into
            # function/module globals or unrelated application object graphs.
            import types

            if not isinstance(item, (types.ModuleType, types.FunctionType, types.MethodType)):
                pending.extend(vars(item).values())


def has_dvm_records(value):
    """Recognize DVM-owned records even if their manifest was damaged/removed.

    The ``_npu_*`` private fields are only ever attached by the DVM compile
    path (graph assets and persistence markers), so their presence is what
    marks a record as ours. They route trusted records, not authenticate
    untrusted data. A bare NPU device type is deliberately not an ownership
    marker.
    """
    fields = {"_npu_manifest", "_npu_fx_assets", "_npu_persistable", "_npu_uncacheable"}
    return any(fields.intersection(getattr(item, "__dict__", {})) for item in walk_objects(value))


def restorable_backends(entry):
    """Match CompilePackage.install, which skips backends of bypassed code."""
    required = {key for code in entry.dynamo.codes if not code.bypassed for key in code.backend_ids}
    missing = required - set(entry.backends)
    if missing:
        # A *partially* missing backend is the case this selection exists for,
        # so name the segment instead of reporting a bare failure.
        raise PackageCompatibilityError("NPU_BACKEND_MISSING:" + ",".join(sorted(missing)))
    return {key: artifact for key, artifact in entry.backends.items() if key in required}
