"""DVM load/view_load selection based on real tensor strides."""

from __future__ import annotations

from typing import Sequence

import torch
from torch._inductor.virtualized import V

from .op_emitter import load, view_load


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


def codegen_maybe_view_load(
    shape: Sequence,
    stride: Sequence,
    dtype: torch.dtype,
    *,
    view_fusion_level: int,
    is_symbolic: bool,
) -> tuple[str, bool]:
    """Return (expr, skip_cont).

    skip_cont=False means the caller must manually materialize a non-contiguous
    input with .contiguous() before launching the DVM kernel.
    """
    if view_fusion_level == 0:
        return load(shape, dtype), False

    if view_fusion_level == 2:
        return view_load(shape, stride, dtype), True

    if is_symbolic:
        return load(shape, dtype), False

    if stride[-1] == 1 and shape[-1] != 1:
        return view_load(shape, stride, dtype), True

    return load(shape, dtype), False
