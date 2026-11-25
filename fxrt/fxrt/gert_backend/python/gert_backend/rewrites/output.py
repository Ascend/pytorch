"""Output-mapping hooks: build the IR form when the default lowering is not enough.

Static indexing keeps storage views; unproven select geometry becomes a symbolic
length-1 slice plus view. reduce_sum always keeps dims (a false keepdim is a following
view). A mutation-only custom op (``mutates_args``, no returns) has a None FX value;
its custom_call output is synthesized from the mutated input so the IR node stays
a tensor.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.fx import Node

from fxrt.gert_backend.rewrites._common import (
    concrete_int,
    emit_slice,
    example_of,
    int_tuple_node,
    node_output,
    shape_of,
)
from fxrt.gert_backend.rewrites._indexing import (
    emit_select,
    emit_strided_cat,
    getitem_axes,
    getitem_input_vectors,
    slice_axes,
    slice_input_vectors,
    select_axis_index,
)
from fxrt.ir import Op
from fxrt import fx_backend as fxb


def _addmm_scale(node, name: str, positional: int):
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > positional:
        return node.args[positional]
    return 1


def addmm_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Build addmm(mat1, mat2, bias) and keep beta/alpha as IR float attrs.

    aten order is (self, mat1, mat2). The first two IR inputs stay the matmul
    pair so transpose inference matches mm. aclnnAddmm receives (bias, mat1, mat2).
    """
    beta = float(_addmm_scale(node, "beta", 3))
    alpha = float(_addmm_scale(node, "alpha", 4))
    self_tensor, mat1, mat2 = input_nodes[0], input_nodes[1], input_nodes[2]
    result = executor.add_op_node(op, [mat1, mat2, self_tensor], node_output(node, sym_mgr))
    result.set_attr("beta", sym_mgr.from_torch_with_sym(beta))
    result.set_attr("alpha", sym_mgr.from_torch_with_sym(alpha))
    return result


def add_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Keep Add's alpha separate from tensor IO, including the optional out tensor."""
    alpha = node.kwargs.get("alpha", 1)
    if not isinstance(alpha, (int, float)):
        raise ValueError("ACLNN Add requires a constant alpha")
    tensors = list(input_nodes[:2])
    if node.kwargs.get("out") is not None:
        tensors.append(input_nodes[-1])
    result = executor.add_op_node(op, tensors, node_output(node, sym_mgr))
    result.set_attr("alpha", sym_mgr.from_torch_with_sym(float(alpha)))
    result.set_attr("alpha_int", sym_mgr.from_torch_with_sym(int(alpha) if isinstance(alpha, int) else 1))
    return result


def add_out_return_refs(gm, tree):
    """Restore caller object identity only for Add out= returns backed by model feeds."""
    destinations = {n: n.kwargs["out"] for n in gm.graph.nodes
                    if n.op == "call_function" and n.target in (torch.add, torch.ops.aten.add.out)
                    and isinstance(n.kwargs.get("out"), Node)}
    feeds = {n: i for i, n in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")}
    leaves = (tree,) if isinstance(tree, Node) else tree
    refs = []
    for index, leaf in enumerate(leaves):
        target = destinations.get(leaf, leaf if leaf in destinations.values() else None)
        if target in feeds:
            refs.append((index, feeds[target]))
    return refs

# This module intentionally reaches into fx_backend internals for the converter-only
# hook overrides, under the same contract as the other rewrites modules.
# pylint: disable=protected-access


# pylint: disable=unused-argument
def slice_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: ``step != 1`` is a cat of length-1 slices."""
    parsed = slice_axes(node)
    if parsed is None or not input_nodes:
        raise RuntimeError("slice rewrite requires a tensor and a concrete step")
    data, shape, axes = parsed
    del data
    rewritten = emit_strided_cat(node, input_nodes[0], shape, axes, executor, sym_mgr)
    if rewritten is None:
        vectors = slice_input_vectors(node)
        if vectors is not None:
            return emit_slice(executor, sym_mgr, input_nodes[0], vectors[1], vectors[2], example_of(node))
    if rewritten is None:
        raise RuntimeError("slice step rewrite could not build offsets and size")
    return rewritten


# pylint: disable=unused-argument
def getitem_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: step 1 is one Slice; a single strided axis is a cat of slices."""
    if not input_nodes:
        raise RuntimeError("getitem rewrite requires a tensor")
    vectors = getitem_input_vectors(node)
    if vectors is not None:
        parsed = getitem_axes(node)
        rewritten = emit_strided_cat(node, input_nodes[0], parsed[1], parsed[2], executor, sym_mgr)
        if rewritten is None:
            return executor.add_op_node(op, input_nodes, node_output(node, sym_mgr))
        return rewritten
    parsed = getitem_axes(node)
    if parsed is None:
        return executor.add_op_node(op, input_nodes, node_output(node, sym_mgr))
    data, shape, axes = parsed
    del data
    rewritten = emit_strided_cat(node, input_nodes[0], shape, axes, executor, sym_mgr)
    if rewritten is None:
        raise RuntimeError("getitem step rewrite could not build offsets and size")
    return rewritten


# pylint: disable=unused-argument
def select_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Keep static select as a view; use symbolic Slice + view otherwise."""
    if not input_nodes:
        raise RuntimeError("select rewrite requires a tensor")
    rewritten = emit_select(node, input_nodes[0], executor, sym_mgr)
    if rewritten is None:
        shape = shape_of(node.args[0])
        parsed = select_axis_index(node)
        if parsed is None:
            raise ValueError("select requires a ranked tensor and constant dim/index")
        axis, index = parsed
        offsets, size = [0] * len(shape), list(shape)
        offsets[axis] = shape[axis] + index if index < 0 else index
        size[axis] = 1
        result = example_of(node)
        sliced = emit_slice(executor, sym_mgr, input_nodes[0], tuple(offsets), tuple(size),
                            result.unsqueeze(axis))
        return executor.add_op_node(Op.view, [sliced, int_tuple_node(executor, sym_mgr, result.shape)],
                                    node_output(node, sym_mgr))
    return rewritten


def _fx_keepdim(node: Node) -> bool:
    """Read keepdim from the FX node. Absent means false, matching torch.sum."""
    if "keepdim" in node.kwargs:
        return bool(node.kwargs["keepdim"])
    if len(node.args) > 2 and not isinstance(node.args[2], Node):
        return bool(node.args[2])
    return False


def _reduce_dims(node: Node, rank: int) -> Optional[List[int]]:
    """Normalize sum dims. None or an empty list reduces every axis."""
    dim = node.kwargs["dim"] if "dim" in node.kwargs else (node.args[1] if len(node.args) > 1 else None)
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return list(range(rank))
    items = dim if isinstance(dim, (list, tuple)) else (dim,)
    dims: List[int] = []
    for item in items:
        axis = concrete_int(item)
        if axis is None:
            return None
        if axis < 0:
            axis += rank
        dims.append(axis)
    return dims


# pylint: disable=unused-argument
def reduce_sum_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: ReduceSum keeps dims. A false keepdim is a view of that result."""
    if not input_nodes:
        raise RuntimeError("reduce_sum rewrite requires a tensor")
    result = example_of(node)
    if _fx_keepdim(node):
        return executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(result))
    shape = shape_of(node.args[0]) if node.args else None
    if shape is None:
        raise RuntimeError("reduce_sum rewrite requires a tensor")
    dims = _reduce_dims(node, len(shape))
    if dims is None:
        raise RuntimeError("reduce_sum rewrite requires concrete dims")
    kept = result
    for axis in sorted(dims):
        kept = kept.unsqueeze(axis)
    reduced = executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(kept))
    view_size = int_tuple_node(executor, sym_mgr, list(result.shape))
    output = sym_mgr.from_torch_with_sym(result)
    return executor.add_op_node(Op.view, [reduced, view_size], output)


# pylint: disable=unused-argument
def argmax_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: ArgMaxV2 squeezes the reduced axis. A true keepdim is a view of
    that squeezed result, the inverse of the reduce_sum contract."""
    if not input_nodes:
        raise RuntimeError("argmax rewrite requires a tensor")
    result = example_of(node)
    if not _fx_keepdim(node):
        return executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(result))
    dim = node.kwargs["dim"] if "dim" in node.kwargs else (node.args[1] if len(node.args) > 1 else None)
    axis = concrete_int(dim)
    if axis is None:
        raise RuntimeError("argmax keepdim rewrite requires a concrete dim")
    squeezed = result.squeeze(axis)
    reduced = executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(squeezed))
    view_size = int_tuple_node(executor, sym_mgr, list(result.shape))
    output = sym_mgr.from_torch_with_sym(result)
    return executor.add_op_node(Op.view, [reduced, view_size], output)


# pylint: disable=unused-argument
def stack_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: Pack's AutoTiling wrapper rejects axis == rank (appending a
    trailing dim), so that corner lowers as a rank+1 view per input plus a ConcatD on
    the last axis -- the same concat DSL kernel family, one copy, no aclnn. Every other
    dim (negative ones included; the lowering normalizes them) goes to Pack unchanged."""
    tensors = node.args[0] if node.args else None
    if not isinstance(tensors, (list, tuple)) or not tensors or not input_nodes:
        raise RuntimeError("stack rewrite requires a leading tensor list")
    dim = node.kwargs["dim"] if "dim" in node.kwargs else (node.args[1] if len(node.args) > 1 else 0)
    axis = concrete_int(dim)
    shape = shape_of(tensors[0])
    if axis is None or shape is None:
        raise RuntimeError("stack rewrite requires a concrete dim and a ranked input")
    rank = len(shape)
    normalized = axis + rank + 1 if axis < 0 else axis
    if normalized != rank:
        return executor.add_op_node(op, input_nodes, node_output(node, sym_mgr))
    result = example_of(node)
    viewed = []
    dim_node = executor.add_value_node(sym_mgr.from_torch_with_sym(-1))
    for fx_tensor, ir_tensor in zip(tensors, input_nodes):
        example = example_of(fx_tensor).unsqueeze(-1)
        viewed.append(executor.add_op_node(
            Op.unsqueeze_view, [ir_tensor, dim_node], sym_mgr.from_torch_with_sym(example)))
    return executor.add_op_node(Op.cat, viewed + [dim_node], sym_mgr.from_torch_with_sym(result))


# pylint: disable=unused-argument
def reshape_and_cache_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: the cache update returns void, so publish the two cache tensors.

    Path C lowers a node only when it has tensor outputs. key_cache and value_cache are
    inputs 2 and 3; the catalog refs those storages, and the graph return still reads them.
    """
    if len(node.args) < 4:
        raise RuntimeError("reshape_and_cache requires key, value, key_cache and value_cache")
    caches = (example_of(node.args[2]), example_of(node.args[3]))
    if any(getattr(cache, "shape", None) is None for cache in caches):
        raise RuntimeError("reshape_and_cache cache inputs have no example shape")
    return executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(caches))


# pylint: disable=unused-argument
def reshape_and_cache_siso_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: the siso cache update returns void, so publish key_cache.

    Path C lowers a node only when it has a tensor output. key_cache is input 1;
    the catalog ref writes that storage, and the graph return still reads it.
    """
    if len(node.args) < 2:
        raise RuntimeError("reshape_and_cache_siso requires key, key_cache and slot_indices")
    cache = example_of(node.args[1])
    if getattr(cache, "shape", None) is None:
        raise RuntimeError("reshape_and_cache_siso key_cache has no example shape")
    return executor.add_op_node(op, input_nodes, sym_mgr.from_torch_with_sym(cache))
def _mutation_ref_example(node: Node):
    """Synthesize the example value for a mutation-only (out-param) custom op.

    Ops that declare ``mutates_args`` and return nothing have a None FX value.
    The effective output is the mutated input, so the IR node stays a tensor
    and Convert lowers it with ref semantics.
    """
    schema = getattr(node.target, "_schema", None)
    if schema is None or list(schema.returns):
        return None
    try:
        positional = [arg for arg in schema.arguments if not getattr(arg, "kwonly", False)]
        mutated = [
            (index, arg) for index, arg in enumerate(positional)
            if arg.alias_info is not None and arg.alias_info.is_write
            and str(arg.type).startswith("Tensor")
        ]
    except (AttributeError, TypeError):
        return None
    if len(mutated) != 1:
        return None
    index, arg_spec = mutated[0]
    arg = None
    if index < len(node.args):
        arg = node.args[index]
    elif arg_spec.name and node.kwargs and arg_spec.name in node.kwargs:
        arg = node.kwargs[arg_spec.name]
    if not isinstance(arg, Node):
        return None
    return arg.meta.get("example_value", arg.meta.get("val"))


# pylint: disable=unused-argument
def custom_call_output_hook(node, op, input_nodes, executor, sym_mgr):
    """Output-mapping: mutation-only ops synthesize their output from the mutated input.

    Every other custom call keeps the default example-value path.
    """
    example_value = fxb._get_node_meta_value(node)
    if example_value is None:
        example_value = _mutation_ref_example(node)
    output_value = sym_mgr.from_torch_with_sym(example_value)
    return executor.add_op_node(op, input_nodes, output_value)


def register() -> None:
    """Register the output-mapping hooks into fx_backend's tables (idempotent)."""
    fxb.register_output_mapping_hook(Op.slice_view, slice_output_hook)
    fxb.register_output_mapping_hook(Op.narrow_view, slice_output_hook)
    fxb.register_output_mapping_hook(Op.getitem_slice, getitem_output_hook)
    fxb.register_output_mapping_hook(Op.select_view, select_output_hook)
    fxb.register_output_mapping_hook(Op.add, add_output_hook)
    fxb.register_output_mapping_hook(Op.addmm, addmm_output_hook)
    fxb.register_output_mapping_hook(Op.reduce_sum, reduce_sum_output_hook)
    # Same keepdim contract for ReduceMean (keep_dims const 1 on the row).
    fxb.register_output_mapping_hook(Op.reduce_mean, reduce_sum_output_hook)
    # And for ReduceMax (amax): identical keep_dims / view contract.
    fxb.register_output_mapping_hook(Op.amax, reduce_sum_output_hook)
    # ReduceMin (amin) shares the same contract.
    if hasattr(Op, "amin"):
        fxb.register_output_mapping_hook(Op.amin, reduce_sum_output_hook)
    # ArgMaxV2 squeezes natively; the hook re-views a true keepdim.
    if hasattr(Op, "argmax"):
        fxb.register_output_mapping_hook(Op.argmax, argmax_output_hook)
    # Pack rejects axis == rank; that corner is views plus a ConcatD.
    fxb.register_output_mapping_hook(Op.stack, stack_output_hook)
    # Void cache update: the two cache tensors are the node's IR outputs.
    fxb.register_output_mapping_hook(Op.reshape_and_cache, reshape_and_cache_output_hook)
    # Siso writes key_cache only. The runtime Op comes from ops.list.
    if hasattr(Op, "reshape_and_cache_siso"):
        fxb.register_output_mapping_hook(
            Op.reshape_and_cache_siso, reshape_and_cache_siso_output_hook)
    fxb.register_output_mapping_hook(Op.custom_call, custom_call_output_hook)
