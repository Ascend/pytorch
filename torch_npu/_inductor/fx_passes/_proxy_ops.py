"""Proxy ops for FA v3 graph partition.

Each proxy mirrors the schema of the underlying ``npu_fusion_attention_v3`` /
``npu_fusion_attention_grad_v3`` op, but is registered with
``Tag.cudagraph_unsafe`` so that inductor's scheduler partitions any FX node
whose ``target`` was rewritten to it. The kernel transparently forwards to the
original op, so the eager fallback path produced by graph partition runs the
unmodified FA v3 implementation.
"""

__all__ = ["PROXY_TARGETS"]

import torch


_LIB = torch.library.Library("npu", "FRAGMENT")


def _clone_schema_under_new_name(original_overload, new_unqualified_name: str) -> str:
    schema_str = str(original_overload._schema)
    head = original_overload._schema.name
    overload = original_overload._schema.overload_name
    if overload:
        head = f"{head}.{overload}"
    new_head = f"npu::{new_unqualified_name}"
    if overload:
        new_head = f"{new_head}.{overload}"
    if not schema_str.startswith(head):
        raise RuntimeError(
            f"Unexpected schema prefix for {original_overload}: {schema_str!r}"
        )
    return new_head + schema_str[len(head) :]


def _register_proxy(original_overload, new_unqualified_name: str):
    _LIB.define(
        _clone_schema_under_new_name(original_overload, new_unqualified_name),
        tags=[torch._C.Tag.cudagraph_unsafe],
    )
    proxy_overload = getattr(
        getattr(torch.ops.npu, new_unqualified_name),
        original_overload._overloadname,
    )

    def _kernel(*args, **kwargs):
        return original_overload(*args, **kwargs)

    _LIB.impl(proxy_overload.name(), _kernel, "CompositeExplicitAutograd")
    # Fake/meta impl: delegate to the original op so AOT tracing reuses its
    # device-propagation rules (FA v3 inputs mix npu q/k/v with cpu
    # actual_seq_qlen/kvlen tensors, which the default FakeTensor logic rejects).
    torch.library.register_fake(proxy_overload.name(), _kernel, lib=_LIB)
    return proxy_overload


PROXY_TARGETS = {
    torch.ops.npu.npu_fusion_attention_v3.default: _register_proxy(
        torch.ops.npu.npu_fusion_attention_v3.default,
        "npu_fusion_attention_v3_unsafe",
    ),
    torch.ops.npu.npu_fusion_attention_grad_v3.default: _register_proxy(
        torch.ops.npu.npu_fusion_attention_grad_v3.default,
        "npu_fusion_attention_grad_v3_unsafe",
    ),
}
