"""Infer native view layouts with PyTorch, without evaluating symbolic sizes."""

from __future__ import annotations

import operator

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Node, map_arg

from fxrt.gert_backend._layout_legalize import (
    _is_metadata_only_view,
    _is_maybe_copying_view,
    _is_as_strided_node,
    _is_expand_node,
    _is_permute_node,
    _node_tensor_meta,
)


VIEW_LAYOUT_ATTR = "fxrt_native_view_layouts"


def is_native_index_view(node: Node) -> bool:
    """Recognize basic tensor indexing; advanced Tensor indices are not views."""
    if node.op == "call_method":
        return node.target in {"select", "narrow"}
    if node.op != "call_function":
        return False
    if node.target is operator.getitem:
        if len(node.args) != 2:
            return False
        indices = node.args[1] if isinstance(node.args[1], tuple) else (node.args[1],)
        return all((isinstance(index, (int, slice)) and not isinstance(index, bool))
                   or index is None or index is Ellipsis
                   for index in indices)
    if node.target in {torch.narrow, torch.select}:
        return True
    schema = getattr(node.target, "_schema", None)
    return schema is not None and schema.name in {"aten::slice", "aten::narrow", "aten::select"}


def _metadata_argument(node):
    value = _node_tensor_meta(node)
    if value is None:
        raise ValueError(f"missing metadata for view argument {node.name}")
    return value


def _evaluate(node, args, kwargs):
    if node.op == "call_method":
        return getattr(args[0], node.target)(*args[1:], **kwargs)
    return node.target(*args, **kwargs)


def _layout_record(source, result):
    if not isinstance(result, FakeTensor):
        raise TypeError("native view inference must return a FakeTensor")
    aliases = torch._C._is_alias_of(source, result)  # pylint: disable=protected-access
    # Relative view operations retain the current allocation's live origin.
    # Explicit as_strided has absolute storage coordinates and uses its own ABI.
    relative_offset = result.storage_offset() - source.storage_offset() if aliases else 0
    return (tuple(source.shape), tuple(source.stride()), tuple(result.shape),
            tuple(result.stride()), relative_offset, aliases, result.is_contiguous())


def _infer_layout_candidates(node: Node, input_candidates=()):
    """Return original-input and dense-input layout candidates for a native view.

    The lowering selects a candidate only when its actual input layout matches.
    This prevents eager's preserved compute-output format from reinterpreting a
    dense allocation written by the converter. Custom operators use their own
    validated layout contract; their fake implementation is not a native view.
    """
    native_view = (_is_metadata_only_view(node) or _is_maybe_copying_view(node) or
                   _is_expand_node(node) or _is_permute_node(node) or
                   is_native_index_view(node) or
                   (node.op == "call_function" and
                    node.target in {torch.squeeze, torch.unsqueeze, torch.select, torch.narrow}))
    if (_is_as_strided_node(node) or not native_view or
            not node.args or not isinstance(node.args[0], Node)):
        return None, ()
    source = _node_tensor_meta(node.args[0])
    if not isinstance(source, FakeTensor):
        return None, ()
    args = map_arg(node.args, _metadata_argument)
    kwargs = map_arg(node.kwargs, _metadata_argument)
    with source.fake_mode:
        dense = torch.empty(source.shape, dtype=source.dtype, device=source.device)
        records = []
        results = []
        seen = set()
        # The first two records retain the original/dense input convention.
        # Further candidates compose metadata views of lowered dense producers.
        for index, candidate in enumerate((source, dense, *input_candidates)):
            if index > 1 and candidate.fake_mode is not source.fake_mode:
                # make_fx's real tracing mode can attach per-node fake modes.
                # Relative native views need geometry, not that mode's storage.
                candidate = torch.empty_strided(candidate.shape, candidate.stride(),
                                                dtype=source.dtype, device=source.device)
            key = (tuple(map(str, candidate.shape)), tuple(map(str, candidate.stride())),
                   str(candidate.storage_offset()))
            if index > 1 and key in seen:
                continue
            candidate_args = (candidate,) + args[1:]
            try:
                result = _evaluate(node, candidate_args, kwargs)
            except RuntimeError:
                if index < 2:
                    raise
                # A strict view may not apply to an alternate physical layout.
                # Do not fabricate a record; lowering rejects an unmatched input.
                continue
            records.append(_layout_record(candidate, result))
            results.append(result)
            seen.add(key)
    return tuple(records), tuple(results)


def infer_native_view_layouts(node: Node):
    """Infer the original and dense input candidates for one native view."""
    return _infer_layout_candidates(node)[0]


def infer_native_view_graph_layouts(gm, excluded=()):
    """Infer layouts before exporting inputs, composing alternate view chains.

    FakeTensor replay may specialize a symbolic size. Finish all replay before
    the IR serializer reads size/stride expressions, so input and view records
    use the same ShapeEnv substitutions. Real storage is never read here.
    """
    layouts = {}
    results = {}
    for node in gm.graph.nodes:
        if node in excluded or node.op not in {"call_function", "call_method"}:
            continue
        candidates = results.get(node.args[0], ()) if node.args and isinstance(node.args[0], Node) else ()
        records, outputs = _infer_layout_candidates(node, candidates)
        if records is not None:
            layouts[node] = records
            results[node] = outputs
    return layouts
