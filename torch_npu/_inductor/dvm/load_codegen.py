"""DVM load/view_load selection based on real tensor strides."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch._inductor.ir import FlexibleLayout
from torch._inductor.virtualized import V

from .op_emitter import load, view_load


def _hint_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    sizevars = getattr(getattr(V, "graph", None), "sizevars", None)
    if sizevars is not None:
        return int(sizevars.size_hint(value))
    return int(value)


def _hint_int_tuple(values: Sequence) -> tuple[int, ...]:
    return tuple(_hint_int(v) for v in values)


def strides_match_contiguous(size: Sequence, stride: Sequence) -> bool:
    if len(size) != len(stride):
        return False
    return _hint_int_tuple(stride) == _hint_int_tuple(
        FlexibleLayout.contiguous_strides(list(size))
    )


def _codegen_dim(value: Any) -> int:
    if isinstance(value, torch.SymInt):
        return -1
    return _hint_int(value)


def _codegen_shape_stride(
    shape: Sequence, stride: Sequence
) -> tuple[list, list]:
    codegen_shape = [_codegen_dim(s) for s in shape]
    codegen_stride = [_codegen_dim(s) for s in stride]
    return codegen_shape, codegen_stride


def patch_gm_placeholder_strides_from_codegen_args(
    gm: torch.fx.GraphModule,
    arg_names: Sequence[str],
) -> None:
    """Patch placeholder meta['val'] with Inductor buffer layout strides at codegen time."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, name in zip(placeholders, arg_names):
        val = node.meta.get("val")
        if isinstance(val, (torch.SymInt, torch.SymFloat)):
            continue
        if not isinstance(val, torch.Tensor):
            continue
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            continue
        layout_stride = tuple(buf.get_stride())
        if layout_stride == tuple(val.stride()):
            continue
        node.meta["val"] = torch.empty_strided(
            val.shape,
            layout_stride,
            dtype=val.dtype,
            device=val.device,
        )


def choose_load_codegen(
    shape: Sequence,
    stride: Sequence,
    dtype: torch.dtype,
    *,
    use_view: bool,
    is_symbolic: bool,
) -> tuple[str, bool]:
    """Return (codegen_expr, cont_flag) for a DVM graph input."""
    codegen_shape, codegen_stride = _codegen_shape_stride(shape, stride)
    if strides_match_contiguous(shape, stride):
        return load(codegen_shape, dtype), True

    if is_symbolic or not use_view:
        return load(codegen_shape, dtype), False

    if (
        len(codegen_stride) >= 1
        and codegen_stride[-1] == 1
        and (len(codegen_shape) == 0 or codegen_shape[-1] != 1)
    ):
        return view_load(codegen_shape, codegen_stride, dtype), True

    return load(codegen_shape, dtype), False
