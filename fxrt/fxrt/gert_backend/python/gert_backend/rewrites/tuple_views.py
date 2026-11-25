"""Normalize fixed-arity split/chunk/unbind to slice/view geometry.

Keep genuine view metadata. Lowering derives geometry from its actual source
and the planner rebases contiguous subviews or writes non-contiguous values.
Dynamic arity keeps the original call instead of freezing example lengths.
"""

from __future__ import annotations

import operator

import torch
from torch.fx import Node, map_arg

from fxrt.gert_backend.rewrites._common import example_of
from fxrt.gert_backend._layout_legalize import _lift_int_or_symint_list
from fxrt.gert_backend.rewrites._framework import register_converter_graph_rewrite


_SPLIT_TARGETS = {
    torch.split, torch.chunk, torch.ops.aten.split.Tensor,
    torch.ops.aten.split.sizes, torch.ops.aten.split.default,
    torch.ops.aten.split_with_sizes.default, torch.ops.aten.chunk.default,
}
_UNBIND_TARGETS = {torch.unbind, torch.ops.aten.unbind.int}


def _has_alias_write(producer):
    seen = set()
    pending = list(producer.users)
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if user.op == "call_method" and str(user.target).endswith("_"):
            return True
        if user.target == operator.setitem:
            return True
        schema = getattr(user.target, "_schema", None)
        if schema is not None and any(arg.alias_info is not None and arg.alias_info.is_write
                                      for arg in schema.arguments):
            return True
        alias_result = (schema is not None and any(result.alias_info is not None for result in schema.returns))
        metadata_method = (user.op == "call_method" and user.target in {
            "view", "reshape", "flatten", "squeeze", "unsqueeze", "t", "transpose", "permute", "narrow", "select"})
        if user.target == operator.getitem or alias_result or metadata_method:
            pending.extend(user.users)
    return False


def _constant(value):
    value = example_of(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, torch.SymInt) and not value.node.expr.free_symbols:
        return int(value.node.expr)
    return None


def normalize_tuple_views_(gm):
    changed = False
    for node in list(gm.graph.nodes):
        unbind = ((node.op == "call_function" and node.target in _UNBIND_TARGETS)
                  or (node.op == "call_method" and node.target == "unbind"))
        split = ((node.op == "call_function" and node.target in _SPLIT_TARGETS)
                 or (node.op == "call_method" and node.target in {"split", "chunk"}))
        if not unbind and not split:
            continue
        # Root/sibling aliases can also be mutated after the split. Walk from
        # tensor feeds as well, so replacing the split with snapshots cannot
        # hide a later write through the original input or one of its views.
        if (_has_alias_write(node)
                or any(_has_alias_write(feed) for feed in gm.graph.nodes if feed.op == "placeholder")):
            raise RuntimeError("fx_converter: multi-output view alias mutation requires a storage-sharing implementation")
        if not node.args or not isinstance(node.args[0], Node):
            continue
        source = node.args[0]
        input_value = example_of(source)
        outputs = example_of(node)
        if not isinstance(input_value, torch.Tensor) or not isinstance(outputs, (tuple, list)):
            continue
        if not outputs or not all(isinstance(value, torch.Tensor) for value in outputs):
            continue
        dim_pos = 1 if unbind else 2
        dim = _constant(node.kwargs.get("dim", node.args[dim_pos] if len(node.args) > dim_pos else 0))
        if dim is None or input_value.dim() == 0 or not -input_value.dim() <= dim < input_value.dim():
            continue
        dim %= input_value.dim()
        extent = input_value.shape[dim]
        lengths = [1 if unbind else value.shape[dim] for value in outputs]
        if any(not isinstance(length, (int, torch.SymInt)) for length in lengths):
            continue
        # Dynamo fixes the tuple arity for this FX graph. Each slice length can
        # still be symbolic; keep those expressions instead of freezing the
        # sample extent or requiring every split dimension to be static.
        constant_extent = _constant(extent)
        constant_lengths = [_constant(length) for length in lengths]
        if (constant_extent is not None and all(length is not None for length in constant_lengths)
                and sum(constant_lengths) != constant_extent):
            continue
        # Replacing a tuple is safe only for the ordinary tuple projection,
        # direct return and tuple-valued argument forms represented by FX.
        pieces = []
        offset = 0
        with gm.graph.inserting_before(node):
            for value, length in zip(outputs, lengths):
                start, stop = _lift_int_or_symint_list(gm.graph, [offset, offset + length])
                sliced = gm.graph.call_function(torch.ops.aten.slice.Tensor,
                                                (source, dim, start, stop, 1))
                # Preserve alias geometry; the selected lowering decides
                # whether the slice shares storage or writes a new value.
                slice_value = value.unsqueeze(dim) if unbind else value
                sliced.meta = {"val": slice_value, "example_value": slice_value}
                piece = sliced
                if unbind:
                    piece = gm.graph.call_function(torch.ops.aten.view.default, (sliced, tuple(value.shape)))
                    piece.meta = {"val": value, "example_value": value}
                pieces.append(piece)
                offset += length
        for user in list(node.users):
            if (user.op == "call_function" and user.target == operator.getitem
                    and len(user.args) == 2 and isinstance(user.args[1], int)):
                user.replace_all_uses_with(pieces[user.args[1]])
                gm.graph.erase_node(user)
            else:
                def replace(item):
                    return tuple(pieces) if item is node else item

                user.args = map_arg(user.args, replace)
                user.kwargs = map_arg(user.kwargs, replace)
        gm.graph.erase_node(node)
        changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


register_converter_graph_rewrite(normalize_tuple_views_, "normalize_tuple_views")
