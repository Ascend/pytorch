"""Converter-only op mappings: torch targets the converter lowers but fx_backend does not.

Kept out of fx_backend.py so its source and flow stay unchanged. ``register`` overlays
the extra ``torch target -> Op`` rows onto fx_backend's op map and marks ``_unsafe_view``
as a view-switch name; both are idempotent dict updates that stay for the process.
"""

from __future__ import annotations

import torch

from fxrt.ir import Op
from fxrt import fx_backend as fxb

# This module intentionally reaches into fx_backend internals to apply converter-only op
# mappings without modifying fx_backend.py source.
# pylint: disable=protected-access

# Converter-only op mappings that are not registered in fx_backend.py
# so that the fx_backend source code and flow stay unchanged.
OP_MAP_EXTRA = {
    torch.empty_strided: Op.empty_strided,
    torch.zeros: Op.zeros,
    torch.ops.aten.zeros.default: Op.zeros,
    torch.full: Op.zeros,
    torch.ops.aten.full.default: Op.zeros,
    torch.ops.aten.add.out: Op.add,
    torch.addmm: Op.addmm,
    torch.ops.aten.addmm: Op.addmm,
    torch.ops.aten.addmm.default: Op.addmm,
    torch.bitwise_not: Op.bitwise_not,
    torch.ops.aten.bitwise_not: Op.bitwise_not,
    torch.ops.aten.bitwise_not.default: Op.bitwise_not,
    torch.tanh: Op.tanh,
    torch.ops.aten.tanh.default: Op.tanh,
    torch.ops.aten.tanh: Op.tanh,
    "tanh": Op.tanh,
    torch.gather: Op.gather,
    torch.ops.aten.index_select: Op.index_select,
    torch.ops.aten.index_select.default: Op.index_select,
    torch.ops.aten._unsafe_view.default: Op.view,
    torch.ops.aten._unsafe_view: Op.view,
    torch.ops.aten.repeat.default: Op.repeat,
    torch.ops.aten.constant_pad_nd.default: Op.constant_pad_nd,
    torch.ops.aten.constant_pad_nd: Op.constant_pad_nd,
    torch._C._nn.pad: Op.constant_pad_nd,
    torch.ops.aten.pad.default: Op.constant_pad_nd,
    torch.ops.aten.pad: Op.constant_pad_nd,
    torch.ops.aten.full_like.default: Op.full_like,
    torch.ops.aten.full_like: Op.full_like,
    torch.full_like: Op.full_like,
    torch.ops.aten.gather: Op.gather,
    torch.ops.aten.gather.default: Op.gather,
    torch.ops.aten.ge.Scalar: Op.ge_scalar,
    torch.ops.aten.ge.Tensor: Op.ge,
    torch.ops.aten.ge: Op.ge,
    torch.ops.aten.ne.Scalar: Op.ne,
    torch.ops.aten.ne.Tensor: Op.ne,
    torch.ops.aten.ne: Op.ne,
    torch.ops.aten.mul.Scalar: Op.mul_scalar,
    torch.ops.aten.mul.Scalar_out: Op.mul_scalar,
    torch.ops.aten.sum.dim_IntList: Op.reduce_sum,
    torch.ops.aten.sum.default: Op.reduce_sum,
    torch.sum: Op.reduce_sum,
    "_unsafe_view": Op.view,
    "constant_pad_nd": Op.constant_pad_nd,
    "pad": Op.constant_pad_nd,
    "full_like": Op.full_like,
    "gather": Op.gather,
    "ge": Op.ge,
    "ne": Op.ne,
    "mul_scalar": Op.mul_scalar,
    "reduce_sum": Op.reduce_sum,
    "sum": Op.reduce_sum,
    torch.ops.aten.add.Scalar: Op.add_scalar,
    torch.ops.aten.sub.Scalar: Op.sub_scalar,
    torch.ops.aten.lt.Scalar: Op.lt_scalar,
    torch.ops.aten.lt.Tensor: Op.lt,
    # fx_backend maps both le variants to Op.le; the scalar one goes to the
    # LessEqual scalar row here (same split as lt / lt_scalar).
    torch.ops.aten.le.Scalar: Op.le_scalar,
    # AOT lowers these torch callables to aten targets the base map does not cover.
    torch.ops.aten.bitwise_or.Tensor: Op.bitwise_or_tensor,
    torch.ops.aten.bitwise_or: Op.bitwise_or_tensor,
    torch.ops.aten.gt.Scalar: Op.gt,
    torch.ops.aten.gt.Tensor: Op.gt,
    torch.ops.aten.gt: Op.gt,
    torch.ops.aten.leaky_relu.default: Op.leaky_relu,
    torch.ops.aten.leaky_relu: Op.leaky_relu,
    torch.ops.aten.mean.default: Op.reduce_mean,
    torch.ops.aten.mean.dim: Op.reduce_mean,
    torch.ops.aten.mean: Op.reduce_mean,
    torch.ops.aten.gelu.default: Op.gelu,
    torch.ops.aten.gelu: Op.gelu,
    torch.ops.aten.square.default: Op.square,
    torch.ops.aten.square: Op.square,
    torch.ops.aten.remainder.Tensor: Op.remainder_tensor_tensor,
    torch.ops.aten.remainder.Scalar: Op.remainder_tensor_scalar,
    # Dynamo keeps these as python callables; mod_op_hook splits scalar vs tensor.
    torch.remainder: Op.remainder_tensor_tensor,
    torch.ops.aten.cumsum.default: Op.cumsum_ext,
    torch.ops.aten.cumsum: Op.cumsum_ext,
    torch.cumsum: Op.cumsum_ext,
    torch.ops.aten.argsort.default: Op.argsort,
    torch.ops.aten.argsort: Op.argsort,
    # tile/abs/isnan/logical_and/argmax/amin have no fx_backend mapping; overlay them here.
    torch.ops.aten.tile.default: Op.tile,
    torch.ops.aten.tile: Op.tile,
    torch.tile: Op.tile,
    "bitwise_or_tensor": Op.bitwise_or_tensor,
    "gt": Op.gt,
    "leaky_relu": Op.leaky_relu,
    "mean": Op.reduce_mean,
    "reduce_mean": Op.reduce_mean,
    torch.ops.aten.bitwise_and.Tensor: Op.bitwise_and_tensor,
    torch.bitwise_and: Op.bitwise_and_tensor,
    "bitwise_and": Op.bitwise_and_tensor,
    torch.ops.aten.expand_as.default: Op.expand,
    "expand_as": Op.expand,
    torch.ops.aten.to.dtype: Op.cast,
    torch.ops.aten.to.prim_dtype: Op.cast,
    "rms_norm": Op.rms_norm,
}


def _map_reshape_and_cache_siso(extra: dict) -> None:
    """Map the ATB siso packet onto its runtime Op. The packet name does not match the enum."""
    if not hasattr(Op, "reshape_and_cache_siso"):
        return
    atb = getattr(torch.ops, "atb", None)
    packet = getattr(atb, "_npu_reshape_and_cache_siso", None) if atb is not None else None
    if packet is None:
        return
    extra[packet] = Op.reshape_and_cache_siso
    overload = getattr(packet, "default", None)
    if overload is not None:
        extra[overload] = Op.reshape_and_cache_siso


def register() -> None:
    """Overlay the converter-only op mappings onto fx_backend's tables (idempotent)."""
    extra = dict(OP_MAP_EXTRA)
    for name in ("abs", "isnan", "logical_and", "argmax", "amin"):
        op = getattr(Op, name, None)
        if op is not None:
            packet = getattr(torch.ops.aten, name)
            extra.update({packet.default: op, packet: op, getattr(torch, name): op})
    _map_reshape_and_cache_siso(extra)
    if hasattr(torch.nn.functional, "rms_norm"):
        extra[torch.nn.functional.rms_norm] = Op.rms_norm
    if hasattr(torch, "rms_norm"):
        extra[torch.rms_norm] = Op.rms_norm
    fxb._OP_MAP.update(extra)
    fxb._VIEW_OP_SWITCH_NAMES.setdefault(Op.view, frozenset())
    fxb._VIEW_OP_SWITCH_NAMES[Op.view] = fxb._VIEW_OP_SWITCH_NAMES[Op.view] | frozenset({"_unsafe_view"})
