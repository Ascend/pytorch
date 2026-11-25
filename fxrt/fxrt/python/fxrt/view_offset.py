"""Storage-offset semantics shared by wrapper producers and converter entry.

The public as_strided API uses storage-absolute offsets. Inductor's wrapper
also uses that spelling for offsets relative to the immediate buffer. Classify
at creation, before passes can discard source metadata; never infer an origin
from a sampled FakeTensor offset.
"""

from enum import IntEnum
import operator

import torch
from torch.fx import Node


class ViewOffsetSemantics(IntEnum):
    """Wire values shared with lowering/common/storage_geometry.h."""

    ABSOLUTE = 0
    IMMEDIATE_RELATIVE = 1
    FEED_RELATIVE = 2
    INHERIT = 3


_ABSOLUTE = "_fxrt_as_strided_absolute_offset"
_RELATIVE = "_fxrt_as_strided_immediate_relative_offset"
_FEED_RELATIVE = "_fxrt_compiled_kernel_relative_offset"
OFFSET_SEMANTICS_ATTR = "fxrt_as_strided_offset_semantics"

_AS_STRIDED_TARGETS = {torch.as_strided, torch.ops.aten.as_strided, torch.ops.aten.as_strided.default}
_VIEW_METHODS = {
    "view", "reshape", "flatten", "unsqueeze", "squeeze", "permute", "transpose",
    "t", "expand", "expand_as", "detach", "alias", "as_strided",
    "slice", "select", "narrow",
}
_VIEW_TARGETS = {
    torch.ops.aten.expand.default, torch.ops.aten.slice.Tensor,
    torch.ops.aten.reshape.default, torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,  # pylint: disable=protected-access
    torch.ops.aten.unsqueeze.default, torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim, torch.ops.aten.squeeze.dims,
    torch.ops.aten.permute.default, torch.ops.aten.transpose.int,
    torch.ops.aten.t.default, torch.ops.aten.select.int, torch.ops.aten.narrow.default,
    torch.ops.aten.alias.default, torch.ops.aten.detach.default,
    torch.reshape, torch.flatten, torch.unsqueeze, torch.squeeze, torch.permute,
    torch.transpose, torch.t, torch.narrow, torch.select,
} | _AS_STRIDED_TARGETS
_FRESH_TARGETS = {
    torch.empty, torch.empty_strided,
    torch.ops.aten.empty.memory_format, torch.ops.aten.empty_strided, torch.ops.aten.empty_strided.default,
}
# These operations preserve a zero origin, whether reshape/flatten aliases or
# allocates. Slice/select/narrow are deliberately excluded, even on a fresh root.
_ZERO_ORIGIN_TARGETS = _VIEW_TARGETS - _AS_STRIDED_TARGETS - {
    torch.ops.aten.slice.Tensor, torch.ops.aten.select.int,
    torch.ops.aten.narrow.default, torch.narrow, torch.select,
}
_ZERO_ORIGIN_METHODS = _VIEW_METHODS - {"as_strided", "slice", "select", "narrow"}


def is_as_strided(node):
    return (node.op == "call_function" and node.target in _AS_STRIDED_TARGETS
            or node.op == "call_method" and node.target == "as_strided")


def explicit_offset(node):
    """Return the argument, retaining symbolic values and keyword spelling."""
    return node.args[3] if len(node.args) > 3 else node.kwargs.get("storage_offset")


def set_view_offset_semantics(node, semantics):
    """Restate a newly created view's contract, dropping stale copied flags."""
    semantics = ViewOffsetSemantics(semantics)
    for key in (_ABSOLUTE, _RELATIVE, _FEED_RELATIVE):
        node.meta.pop(key, None)
    if semantics == ViewOffsetSemantics.IMMEDIATE_RELATIVE:
        node.meta[_RELATIVE] = True
    elif semantics in (ViewOffsetSemantics.ABSOLUTE, ViewOffsetSemantics.FEED_RELATIVE):
        node.meta[_ABSOLUTE] = True
        if semantics == ViewOffsetSemantics.FEED_RELATIVE:
            node.meta[_FEED_RELATIVE] = True
    return node


def view_offset_semantics(node):
    """Read producer stamps. Unstamped calls retain the public API contract."""
    absolute = bool(node.meta.get(_ABSOLUTE, False))
    relative = bool(node.meta.get(_RELATIVE, False))
    feed_relative = bool(node.meta.get(_FEED_RELATIVE, False))
    if relative and (absolute or feed_relative):
        raise ValueError(f"as_strided offset provenance is conflicting for node {node.format_node()}")
    if explicit_offset(node) is None:
        return ViewOffsetSemantics.INHERIT
    if relative:
        return ViewOffsetSemantics.IMMEDIATE_RELATIVE
    if feed_relative:
        return ViewOffsetSemantics.FEED_RELATIVE
    return ViewOffsetSemantics.ABSOLUTE


def _source_targets(node):
    return [entry[1] for entry in node.meta.get("source_fn_stack", ())
            if isinstance(entry, tuple) and len(entry) > 1]


def _is_storage_view(node):
    if node.op == "call_function" and node.target is operator.getitem and len(node.args) == 2:
        indices = node.args[1] if isinstance(node.args[1], tuple) else (node.args[1],)
        # Advanced tensor indexing allocates; basic indexing preserves storage.
        return all((isinstance(i, (int, slice)) and not isinstance(i, bool))
                   or i is None or i is Ellipsis for i in indices)
    return (node.op == "call_function" and node.target in _VIEW_TARGETS
            or node.op == "call_method" and node.target in _VIEW_METHODS)


def has_absolute_as_strided_offset(node):
    """Whether a view chain terminates live feed-origin inheritance."""
    while isinstance(node, Node) and node.args:
        if is_as_strided(node) and explicit_offset(node) is not None:
            semantics = view_offset_semantics(node)
            if semantics == ViewOffsetSemantics.FEED_RELATIVE:
                return False
            if semantics == ViewOffsetSemantics.ABSOLUTE and (
                    node.meta.get(_ABSOLUTE, False) or node.target in _AS_STRIDED_TARGETS - {torch.as_strided}
                    or node.op == "call_method" or any(t in _AS_STRIDED_TARGETS for t in _source_targets(node))):
                return True
        if not _is_storage_view(node):
            break
        node = node.args[0]
    return False


def has_zero_storage_origin(node):
    """Prove the immediate origin is zero without consulting tensor metadata."""
    while isinstance(node, Node):
        if node.op == "call_function" and node.target in _FRESH_TARGETS:
            return node.kwargs.get("out") is None
        if not node.args or not (
                node.op == "call_function" and node.target in _ZERO_ORIGIN_TARGETS
                or node.op == "call_method" and node.target in _ZERO_ORIGIN_METHODS):
            return False
        node = node.args[0]
    return False


def classify_inductor_view_offset(node):
    """Classify a host view or reject ambiguous provenance before conversion.

    ATen and method calls retain their public contract unless a producer stamps
    them otherwise. The synthetic torch.as_strided spelling requires producer
    evidence, a recognized source, or a structural proof of equivalent origins.
    """
    if not is_as_strided(node):
        return
    semantics = view_offset_semantics(node)  # Validate even omitted offsets.
    if explicit_offset(node) is None:
        return
    if any(node.meta.get(key, False) for key in (_ABSOLUTE, _RELATIVE, _FEED_RELATIVE)):
        return
    targets = _source_targets(node)
    if (node.op == "call_method" or node.target in _AS_STRIDED_TARGETS - {torch.as_strided}
            or any(t in _AS_STRIDED_TARGETS for t in targets)):
        semantics = ViewOffsetSemantics.ABSOLUTE
    elif any(t in _VIEW_TARGETS - _AS_STRIDED_TARGETS or t in _VIEW_METHODS - {"as_strided"}
             for t in targets):
        semantics = ViewOffsetSemantics.IMMEDIATE_RELATIVE
    elif node.args and has_zero_storage_origin(node.args[0]):
        # This is an equivalence proof for any valid constant/symbolic offset,
        # not a claim that a lost producer was necessarily Inductor's wrapper.
        semantics = ViewOffsetSemantics.IMMEDIATE_RELATIVE
    else:
        raise ValueError(
            f"as_strided offset provenance is unknown for node {node.format_node()}; "
            "preserve source_fn_stack or mark the node as an Inductor-generated view"
        )
    set_view_offset_semantics(node, semantics)
