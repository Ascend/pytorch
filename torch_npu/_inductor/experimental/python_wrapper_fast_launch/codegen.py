from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .backend import arg_kind_from_abi_signature


def _runtime_signatures(triton_meta: Any, arg_count: int) -> tuple[str, ...]:
    if not isinstance(triton_meta, Mapping):
        return ()
    signature = triton_meta.get("signature")
    if not isinstance(signature, Mapping):
        return ()
    constants = triton_meta.get("constants", {}) or {}
    constant_names = {str(name) for name in constants}
    constant_indices = {
        int(index)
        for index in constants
        if isinstance(index, int) or str(index).isdigit()
    }
    values = tuple(
        str(value)
        for index, (name, value) in enumerate(signature.items())
        if str(name) not in constant_names
        and index not in constant_indices
        and str(value) != "constexpr"
    )
    if len(values) == arg_count:
        return values
    all_values = tuple(str(value) for value in signature.values())
    return all_values if len(all_values) == arg_count else ()


def build_callsite_metadata(
    *,
    kernel_name: str,
    call_args: Any,
    triton_meta: Any,
    graph_id: str,
    callsite_index: int,
) -> dict[str, Any]:
    arg_exprs = tuple(str(arg) for arg in call_args)
    signatures = _runtime_signatures(triton_meta, len(arg_exprs))
    arg_kinds = tuple(arg_kind_from_abi_signature(value) for value in signatures)
    schema_state = "complete"
    schema_reason = None
    if not signatures:
        schema_state = "incomplete"
        schema_reason = "codegen_signature_missing"
    elif any(kind is None for kind in arg_kinds):
        schema_state = "incomplete"
        schema_reason = "codegen_arg_kind_unsupported"
    schema_payload = {
        "kernel_name": kernel_name,
        "arg_exprs": arg_exprs,
        "arg_signatures": signatures,
        "arg_kinds": arg_kinds,
        "schema_state": schema_state,
    }
    schema_hash = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    callsite_id = f"{graph_id}:{callsite_index}"
    return {
        "graph_id": graph_id,
        "callsite_id": callsite_id,
        "callsite_index": callsite_index,
        "kernel_name": kernel_name,
        "schema_hash": schema_hash,
        "runtime_arg_count": len(arg_exprs),
        "arg_kinds": tuple(kind for kind in arg_kinds if kind is not None),
        # An incomplete codegen schema is only a hint. The selected launcher
        # owns the final ABI, including runtime block arguments, so promotion
        # may safely complete the schema after autotuning has stabilized.
        "schema_state": schema_state,
        "schema_reason": schema_reason,
        "eligible": True,
        "fallback_reason": None,
    }


__all__ = ["build_callsite_metadata"]
