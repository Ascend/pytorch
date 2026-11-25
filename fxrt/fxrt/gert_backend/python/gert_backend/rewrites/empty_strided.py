"""Represent non-contiguous allocations as physical storage plus a no-copy view.

Compiled kernels address buffers using their original strides. Normalize before
the compiled-kernel adapter restores those layouts, so it can find a contiguous
allocation root instead of rejecting a directly produced non-contiguous buffer.
"""

import torch
from torch._prims_common import compute_elementwise_output_logical_to_physical_perm

from fxrt.compiled_kernel_adapter import _layout_is_plain
from fxrt.gert_backend.rewrites._common import example_of
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


_TARGETS = {torch.empty_strided, torch.ops.aten.empty_strided.default, torch.ops.aten.empty_strided}


def _physical_allocation(value, span):
    """Keep proven physical axes, so transpose-aware consumers can read directly."""
    permutation = compute_elementwise_output_logical_to_physical_perm(value)
    # PyTorch added an ambiguity result alongside the permutation.
    if isinstance(permutation, tuple) and len(permutation) == 2 and isinstance(permutation[0], list):
        permutation = permutation[0]
    physical = value.permute(permutation)
    # Sorting only proposes a candidate. Exact symbolic equality must prove it
    # is dense; padded/overlapping layouts retain their full flat storage span.
    if _layout_is_plain(physical):
        return physical
    return value.as_strided((span,), (1,), 0)


def normalize_empty_strided_(gm):
    """Allocate the full storage span, retaining symbolic sizes and strides."""
    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _TARGETS:
            continue
        value = example_of(node)
        if not isinstance(value, torch.Tensor) or _layout_is_plain(value):
            continue
        if any(isinstance(size, int) and size == 0 for size in value.shape):
            continue

        # PyTorch has already validated the requested layout and computed its
        # allocation span. Reuse that span, including holes; numel is too small
        # for padded strides. This operation only reads FakeTensor metadata.
        span = value.untyped_storage().nbytes() // value.element_size()
        storage_value = _physical_allocation(value, span)
        extents = [*storage_value.shape, *storage_value.stride(), *value.shape, *value.stride()]
        with gm.graph.inserting_before(node):
            if any(isinstance(extent, torch.SymInt) for extent in extents):
                materialize = getattr(gm.graph, "materialize_symints", None)
                if materialize is None:
                    raise RuntimeError("fx_converter: symbolic empty_strided requires FX SymInt materialization")
                extents = materialize(extents)
            rank = value.dim()
            storage_rank = storage_value.dim()
            logical_start = 2 * storage_rank
            kwargs = {key: val for key, val in node.kwargs.items() if key not in {"size", "stride"}}
            storage = gm.graph.call_function(
                node.target,
                (extents[:storage_rank], extents[storage_rank:logical_start]),
                kwargs,
            )
            storage.meta = {**node.meta, "val": storage_value, "example_value": storage_value}
            storage.meta.pop("tensor_meta", None)
            view = gm.graph.call_function(torch.ops.aten.as_strided.default,
                                          (storage, extents[logical_start:logical_start + rank],
                                           extents[logical_start + rank:], 0))
            view.meta = dict(node.meta)
        node.replace_all_uses_with(view)
        gm.graph.erase_node(node)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()


register_converter_graph_rewrite(normalize_empty_strided_, "normalize_empty_strided")
