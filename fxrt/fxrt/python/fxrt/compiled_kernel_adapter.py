"""Compiled-kernel HOP adapter for the torch.fx backend.

The fx_wrapper keeps a fusion backend's fused kernels alive in the inductor host
graph as ``compiled_kernel_wrapper_mutation`` HOP nodes (see
:mod:`fxrt.compiled_kernel_hop`). This module owns the shared node
helpers and the generic lowering of such a node to the FXRT
``compiled_kernel_mutation`` op, which launches the kernel on the buffers the
graph already holds -- including graph inputs the kernel mutates in place.

A fusion backend whose kernels FXRT can run natively lowers them to its own op
instead -- see :func:`fxrt.dvm_adapter.lower_compiled_kernel_dvm_node`,
which the fx backend tries first.

A compiled kernel's index arithmetic is frozen at codegen time against the
layout its FX node carries, so this module also owns the graph pass that pins
that layout down -- see :func:`restore_compiled_kernel_arg_layouts_` -- and the
pass that turns a permuted allocation FXRT cannot serve into plain memory plus a
view, see :func:`normalize_fresh_allocation_layouts_`.
"""

import importlib
import os
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Node

from fxrt.ir import Op
from fxrt.view_offset import (
    ViewOffsetSemantics, has_absolute_as_strided_offset, set_view_offset_semantics,
)

_HOP_NAME = "compiled_kernel_wrapper_mutation"
_HOP_MODULE = "fxrt.compiled_kernel_hop"


def is_compiled_kernel_wrapper_mutation_target(target) -> bool:
    """Return whether an FX node target is the compiled-kernel mutation HOP."""
    target_name = getattr(target, "__name__", None)
    if target_name == _HOP_NAME:
        return True
    return str(target) == f"torch.ops.higher_order.{_HOP_NAME}"


def get_compiled_kernel_index(node: Node) -> int:
    """Extract the side-table index of the compiled kernel a HOP node calls."""
    kernel_idx = node.kwargs.get("kernel_idx", None)
    if not isinstance(kernel_idx, int):
        raise RuntimeError(f"{_HOP_NAME} requires integer kwargs['kernel_idx']")
    return kernel_idx


def resolve_compiled_kernel(kernel_idx: int):
    """Look up a compiled kernel from the fx_wrapper side table."""
    try:
        module = importlib.import_module(_HOP_MODULE)
    except ImportError as exc:
        raise RuntimeError(f"{_HOP_NAME} requires {_HOP_MODULE}") from exc
    return module.compiled_kernel_side_table.get_kernel(kernel_idx)


def get_compiled_kernel_mutation_args(node: Node) -> Tuple[Tuple[Argument, ...], Tuple[int, ...]]:
    """Extract explicit args and mutated output indices from a mutation HOP node."""
    if not is_compiled_kernel_wrapper_mutation_target(node.target):
        raise RuntimeError(f"Expected {_HOP_NAME} node")

    hop_args = node.kwargs.get("args", None)
    mutated_arg_indices = node.kwargs.get("mutated_arg_indices", ())
    if not isinstance(hop_args, tuple):
        raise RuntimeError(f"{_HOP_NAME} requires tuple kwargs['args']")
    if not isinstance(mutated_arg_indices, tuple) or not all(
        isinstance(index, int) for index in mutated_arg_indices
    ):
        raise RuntimeError(f"{_HOP_NAME} requires tuple[int, ...] kwargs['mutated_arg_indices']")
    return hop_args, mutated_arg_indices


def get_compiled_kernel_mutation_output_nodes(node: Node) -> List[Node]:
    """Return FX nodes that represent mutated output buffers."""
    hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)
    output_nodes = []
    for index in mutated_arg_indices:
        if index < 0 or index >= len(hop_args):
            raise RuntimeError(
                f"{_HOP_NAME} mutated arg index {index} is out of range for {len(hop_args)} args"
            )
        output_node = hop_args[index]
        if not isinstance(output_node, Node):
            raise RuntimeError(
                f"{_HOP_NAME} mutated arg[{index}] must be an FX node, got {type(output_node)}"
            )
        output_nodes.append(output_node)
    return output_nodes


def map_hop_args(args, env, executor, sym_mgr) -> List[Node]:
    """Map compiled-kernel HOP arguments to FXRT graph nodes."""

    def _map_arg(arg):
        if isinstance(arg, Node):
            return env[arg]
        if isinstance(arg, (list, tuple)):
            return executor.make_tuple([_map_arg(item) for item in arg])
        return executor.add_value_node(sym_mgr.from_torch_with_sym(arg))

    return [_map_arg(arg) for arg in args]


def lower_compiled_kernel_node(
        node,
        executor,
        env,
        sym_mgr,
        get_node_meta_value,
        add_tuple_getitem_node,
):
    """Lower a compiled-kernel mutation HOP node into an FXRT compiled_kernel_mutation op.

    Returns False for nodes that are not compiled-kernel HOP calls, so the fx
    backend can keep lowering them the usual way.
    """
    if not is_compiled_kernel_wrapper_mutation_target(node.target):
        return False

    kernel_idx = get_compiled_kernel_index(node)
    # Fail here rather than at run time if the kernel is gone from the side table.
    resolve_compiled_kernel(kernel_idx)

    hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)
    output_nodes = get_compiled_kernel_mutation_output_nodes(node)
    if not output_nodes:
        raise RuntimeError(
            f"{_HOP_NAME} requires at least one mutated output buffer, "
            f"a compiled kernel that writes nothing has no effect"
        )

    output_examples = []
    for output_node in output_nodes:
        example_value = get_node_meta_value(output_node)
        if example_value is None:
            raise RuntimeError(
                f"Compiled kernel output buffer node '{output_node.name}' is missing example_value/val metadata"
            )
        output_examples.append(example_value)

    # Every kernel arg is passed as an input, the mutated ones included: the op refs each output
    # to the buffer it is written into, so the kernel writes the buffer the graph already holds
    # -- a graph input mutated by the kernel reaches the caller, and a buffer the kernel only
    # partially updates keeps what it held on entry.
    # input[0] and input[1] are compile-time constants, so they are value nodes rather than ops.
    input_nodes = [
        executor.add_value_node(sym_mgr.from_torch_with_sym(kernel_idx)),
        executor.add_value_node(sym_mgr.from_torch_with_sym(tuple(mutated_arg_indices))),
    ]
    input_nodes += map_hop_args(hop_args, env, executor, sym_mgr)

    compiled_kernel = resolve_compiled_kernel(kernel_idx)
    metadata_fn = getattr(compiled_kernel, "fxrt_converter_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{_HOP_NAME} backend metadata must be a dict")

    def _attach_metadata(kernel_node):
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise RuntimeError(
                    f"{_HOP_NAME} metadata must contain string key/value pairs"
                )
            kernel_node.set_attr(key, sym_mgr.from_torch_with_sym(value))

    if len(output_examples) == 1:
        output_value = sym_mgr.from_torch_with_sym(output_examples[0])
        kernel_node = executor.add_op_node(Op.compiled_kernel_mutation, input_nodes, output_value)
        _attach_metadata(kernel_node)
        env[node] = kernel_node
        env[output_nodes[0]] = kernel_node
        return True

    tuple_output = sym_mgr.from_torch_with_sym(tuple(output_examples))
    tuple_node = executor.add_op_node(Op.compiled_kernel_mutation, input_nodes, tuple_output)
    _attach_metadata(tuple_node)
    env[node] = tuple_node
    for idx, (output_node, output_example) in enumerate(zip(output_nodes, output_examples)):
        output_value = sym_mgr.from_torch_with_sym(output_example)
        env[output_node] = add_tuple_getitem_node(executor, sym_mgr, tuple_node, idx, output_value)
    return True


# ---------------------------------------------------------------------------
# Argument layout restoration
#
# A compiled kernel indexes each argument with the strides its FX node carries:
# inductor bakes that arithmetic into the kernel at codegen time. FXRT is free
# to pick its own layout for any intermediate -- `aten.expand`, for one, lowers
# to aclnnExpand, which materializes a dense tensor instead of a stride-0 view.
# Values stay correct everywhere else, but a kernel handed that dense buffer
# still addresses it with the broadcast strides and reads the wrong elements, or
# runs off the end of the allocation.
#
# So the layout is spelled out at the kernel boundary: every argument whose node
# asks for something a plain contiguous allocation does not provide is rewritten
# to an explicit as_strided on the buffer its view chain roots at.
# ---------------------------------------------------------------------------

# Storage-preserving ops the base walk may step through. Sharing storage is not
# enough on its own -- an in-place op aliases its input too, and stepping over
# one would drop the write. These are all pure, so a node left unused after a
# rewrite can also be erased.
_PURE_VIEW_TARGETS = frozenset(
    target
    for target in (
        getattr(torch.ops.aten.expand, "default", None),
        getattr(torch.ops.aten.slice, "Tensor", None),
        getattr(torch.ops.aten.reshape, "default", None),
        getattr(torch.ops.aten.view, "default", None),
        getattr(torch.ops.aten._unsafe_view, "default", None),  # pylint: disable=protected-access
        getattr(torch.ops.aten.unsqueeze, "default", None),
        getattr(torch.ops.aten.squeeze, "default", None),
        getattr(torch.ops.aten.squeeze, "dim", None),
        getattr(torch.ops.aten.squeeze, "dims", None),
        getattr(torch.ops.aten.permute, "default", None),
        getattr(torch.ops.aten.transpose, "int", None),
        getattr(torch.ops.aten.t, "default", None),
        getattr(torch.ops.aten.select, "int", None),
        getattr(torch.ops.aten.narrow, "default", None),
        getattr(torch.ops.aten.alias, "default", None),
        getattr(torch.ops.aten.detach, "default", None),
        getattr(torch.ops.aten.as_strided, "default", None),
        torch.as_strided,
    )
    if target is not None
)


def _node_meta_value(node: Node):
    """Return the fake tensor / symint an FX node carries, or None."""
    return node.meta.get("example_value", node.meta.get("val", None))


def _untyped_storage(value):
    """Return a tensor's untyped storage, or None when it cannot be reached."""
    if not isinstance(value, torch.Tensor):
        return None
    try:
        return value.untyped_storage()
    except Exception:  # pylint: disable=broad-exception-caught
        # Some tensor subclasses have no storage to speak of; treat them as
        # unrelated rather than guessing.
        return None


def _shares_storage(candidate, value) -> bool:
    """Whether two fake tensors are backed by the very same storage."""
    candidate_storage = _untyped_storage(candidate)
    return candidate_storage is not None and candidate_storage is _untyped_storage(value)


def _same_extent(lhs, rhs) -> bool:
    """Compare two sizes / strides / offsets without planting a guard."""
    lhs_expr = lhs.node.expr if isinstance(lhs, torch.SymInt) else lhs
    rhs_expr = rhs.node.expr if isinstance(rhs, torch.SymInt) else rhs
    return bool(lhs_expr == rhs_expr)


def _same_extents(lhs, rhs) -> bool:
    """Compare two sizes / strides lists element by element, guard-free."""
    lhs = list(lhs)
    rhs = list(rhs)
    return len(lhs) == len(rhs) and all(
        _same_extent(left, right) for left, right in zip(lhs, rhs)
    )


def _contiguous_strides(sizes: List[Any]) -> List[Any]:
    """Strides a freshly allocated contiguous buffer of this shape would have."""
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return strides


def _layout_is_plain(value: torch.Tensor) -> bool:
    """Whether a plain contiguous allocation addresses `value` identically.

    A dimension of length 1 is always indexed at 0, so its stride never reaches
    the address arithmetic and does not have to match.
    """
    if not _same_extent(value.storage_offset(), 0):
        return False
    sizes = list(value.shape)
    expected = _contiguous_strides(sizes)
    for size, actual, wanted in zip(sizes, value.stride(), expected):
        if isinstance(size, int) and size == 1:
            continue
        if not _same_extent(actual, wanted):
            return False
    return True


_STRIDED_ALLOC_NAMES = frozenset({
    "empty_strided",
    "new_empty_strided",
})

_FRESH_ALLOC_NAMES = frozenset({
    "empty",
    "empty_strided",
    "empty_like",
    "new_empty",
    "new_empty_strided",
    "zeros",
    "zeros_like",
    "full",
    "full_like",
})

# Ops that only re-index the buffer they are handed, so their own metadata can be
# reproduced as an as_strided over that buffer.
_REINDEX_NAMES = frozenset({
    "reshape",
    "view",
    "_unsafe_view",
    "flatten",
    "unflatten",
    "squeeze",
    "unsqueeze",
    "permute",
    "transpose",
    "t",
    "select",
    "narrow",
    "slice",
    "alias",
    "detach",
    "as_strided",
})

# Those of them that materialize a copy whenever the source layout cannot be
# viewed as the requested shape.
_COPYING_REINDEX_NAMES = frozenset({
    "reshape",
    "view",
    "_unsafe_view",
    "flatten",
    "unflatten",
})


def _target_name(target) -> str:
    """Short operator name of an FX call target (torch function or OpOverload)."""
    schema = getattr(target, "_schema", None)
    qualified = getattr(schema, "name", None)
    if isinstance(qualified, str):
        # OpOverload: "aten::reshape", where str(target) would only say "default".
        return qualified.rsplit("::", 1)[-1]
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name.isidentifier():
        return name
    return str(target).rsplit(".", 1)[-1]


def _is_fresh_allocation(node: Node) -> bool:
    """Whether the node allocates the buffer it produces, so nothing else aliases it."""
    if node.op != "call_function":
        return False
    return _target_name(node.target) in _FRESH_ALLOC_NAMES


def _is_dense_permutation(value: torch.Tensor) -> bool:
    """Whether the layout is a dense permutation of a contiguous layout.

    Such a buffer covers exactly ``numel()`` elements, so a plain contiguous
    allocation of the same shape is large enough to be viewed with its strides.
    """
    if not _same_extent(value.storage_offset(), 0):
        return False
    sizes = list(value.shape)
    strides = list(value.stride())
    live = [dim for dim in range(len(sizes)) if not _same_extent(sizes[dim], 1)]
    if not live:
        return True
    for order in permutations(live):
        expected = 1
        for dim in order:
            if not _same_extent(strides[dim], expected):
                break
            expected = expected * sizes[dim]
        else:
            return True
    return False


def _contiguous_layout(value: torch.Tensor) -> bool:
    """Whether a contiguous allocation would address `value` identically.

    Symbolic extents are compared as expressions, so a buffer whose shape is
    only known at run time is recognized as plain instead of being rewritten
    into an allocation plus a redundant view.
    """
    return _layout_is_plain(value)


def _is_strided_empty(target) -> bool:
    """Whether the target spells an allocation's layout out as a stride list."""
    return _target_name(target) == "empty_strided"


def _layout_extents(value: torch.Tensor):
    """The (sizes, strides) a view of `value` has to spell out."""
    return list(value.shape), list(value.stride())


def _extent_arg_value(arg):
    """The size / stride a graph argument stands for."""
    return _node_meta_value(arg) if isinstance(arg, Node) else arg


def _strides_are_canonical(args, strides) -> bool:
    """Whether a node's stored stride arguments already spell `strides` out."""
    if not isinstance(args, (list, tuple)) or len(args) != len(strides):
        return False
    return all(
        _same_extent(_extent_arg_value(arg), stride)
        for arg, stride in zip(args, strides)
    )


def _node_precedes(candidate: Node, consumer: Node) -> bool:
    """Whether `candidate` is defined before `consumer` in the same graph."""
    for node in consumer.graph.nodes:
        if node is consumer:
            return False
        if node is candidate:
            return True
    return False


def _extent_node_before(
    node: Optional[Node],
    consumer: Optional[Node],
    symbols: Optional[Dict[str, Node]] = None,
    key: Optional[str] = None,
) -> Optional[Node]:
    """Return a node producing `node`'s value ahead of `consumer`.

    FX requires arguments to be defined before the node consuming them. A
    symbolic extent that inductor only materialises further down the graph --
    ``32*s64`` first spelled out for a later allocation, say -- therefore cannot
    be handed over as-is.

    The node is moved ahead of the consumer rather than copied: every user is
    repointed at the hoisted node and the original is dropped, so the extent is
    still computed exactly once and hoisting costs the graph no extra work.
    Returns ``None`` when it cannot be hoisted -- one of its inputs is itself
    defined after the consumer -- leaving the caller to pass the SymInt through.
    """
    if node is None or consumer is None:
        return node
    if _node_precedes(node, consumer):
        return node
    for arg in node.all_input_nodes:
        if not _node_precedes(arg, consumer):
            return None
    with consumer.graph.inserting_before(consumer):
        hoisted = consumer.graph.node_copy(node)
    hoisted.meta = dict(node.meta)
    node.replace_all_uses_with(hoisted)
    consumer.graph.erase_node(node)
    if symbols is not None and key is not None:
        symbols[key] = hoisted
    return hoisted


def _layout_args(
    extents, symbols: Dict[str, Node], before: Optional[Node] = None
) -> List[Any]:
    """Spell sizes / strides out as arguments for a graph node.

    A symbolic extent is passed as the node that produces it, which is what an
    FXRT executor resolves; when no node produces that expression the SymInt is
    passed through as-is, which the symbolic shape manager resolves from the
    expression itself.

    `before` is the node the arguments are spelled out for. Passing it keeps the
    arguments defined ahead of their consumer -- see `_extent_node_before`.
    """
    args: List[Any] = []
    for extent in extents:
        if isinstance(extent, torch.SymInt):
            expr = extent.node.expr
            if expr.is_Integer:
                args.append(int(expr))
                continue
            key = str(expr)
            node = _extent_node_before(symbols.get(key), before, symbols, key)
            if node is not None:
                args.append(node)
                continue
        args.append(extent)
    return args


def _plain_meta(value: torch.Tensor, sizes) -> torch.Tensor:
    """Metadata of a freshly allocated contiguous buffer of `sizes`.

    A fake tensor stays fake, so the example describes a buffer of its own
    instead of aliasing the tensor it was derived from. Symbolic extents are
    materialized inside the tensor's own fake mode, since the device dispatcher
    only accepts extents it can turn into an IntArrayRef.
    """
    strides = _contiguous_strides(list(sizes))
    if isinstance(value, FakeTensor):
        with value.fake_mode:
            return torch.empty_strided(
                list(sizes), strides, dtype=value.dtype, device=value.device
            )
    return value.as_strided(list(sizes), strides, 0)


def _set_meta_value(node: Node, value) -> None:
    """Store `value` where the backends look a node's example tensor up."""
    stored = False
    for key in ("example_value", "val"):
        if key in node.meta:
            node.meta[key] = value
            stored = True
    if not stored:
        node.meta["example_value"] = value


def _stamp(node: Node, source: Node, value) -> None:
    """Give a freshly emitted node the metadata of the node it replaces."""
    node.meta.update(source.meta)
    _set_meta_value(node, value)


def _mark_absolute_storage_offset(node: Node) -> Node:
    """Declare that `node`'s offset addresses a root buffer's storage directly.

    The adapter spells out exactly the layout its fake tensor reports: the offset
    is `Tensor.storage_offset()`, i.e. absolute in the storage of a base it
    validated to start at element 0 (or a feed whose storage origin FXRT already
    tracks). Inductor's wrapper, by contrast, emits offsets relative to the
    immediate input and marks those nodes itself; copying the replaced node's
    provenance would therefore misclassify this call. The converter classifies
    host-graph views from this metadata and rejects an unclassified one, so the
    absolute contract has to be restated on every node this adapter emits.
    """
    return set_view_offset_semantics(node, ViewOffsetSemantics.ABSOLUTE)


def _as_strided_on(gm: GraphModule, base: Node, sizes, strides, offset,
                   source: Node, value) -> Node:
    """Emit an `as_strided` over `base` that carries `value` as its example."""
    node = gm.graph.call_function(torch.as_strided, args=(base, sizes, strides, offset))
    _stamp(node, source, value)
    return _mark_absolute_storage_offset(node)


def _reindex_relation(candidate: Node, holder_value: torch.Tensor) -> str:
    """How `candidate` relates to the buffer its producer holds.

    ``"view"`` when the node only re-indexes that buffer, ``"copy"`` when the
    node materializes the data (a reshape of a layout it cannot view), and
    ``""`` when the node is not a pure re-index of it.
    """
    if candidate.op != "call_function" or _target_name(candidate.target) not in _REINDEX_NAMES:
        return ""
    if not candidate.args or not isinstance(candidate.args[0], Node):
        return ""
    value = _node_meta_value(candidate)
    if not isinstance(value, torch.Tensor):
        return ""
    if not isinstance(holder_value, torch.Tensor):
        return ""
    if _shares_storage(value, holder_value):
        return "view"
    if not _same_extent(value.storage_offset(), 0):
        return ""
    if not _same_extent(value.numel(), holder_value.numel()):
        return ""
    name = _target_name(candidate.target)
    if name not in _COPYING_REINDEX_NAMES:
        # permute / transpose / slice / squeeze & co. are views by construction.
        return "view"
    # Inductor rebuilds metadata instead of tracing views, so ask PyTorch what
    # this shape does to the producer's layout.
    try:
        probe = holder_value.reshape(list(value.shape))
    except Exception:  # pylint: disable=broad-exception-caught
        return ""
    if _shares_storage(probe, holder_value):
        return "view" if _same_extents(probe.stride(), value.stride()) else ""
    if (
        _same_extents(probe.shape, value.shape)
        and _same_extents(probe.stride(), _contiguous_strides(list(value.shape)))
        and _same_extent(probe.storage_offset(), 0)
    ):
        return "copy"
    return ""


def _align_views_with_alloc(gm: GraphModule, alloc: Node, holders: List[Node]) -> int:
    """Re-root the views of a permuted buffer on its plain allocation.

    A view of the permuted buffer is folded into an `as_strided` over the plain
    allocation -- identical address arithmetic, without the layout that FXRT's
    View op refuses to read. A reshape that PyTorch would copy stays a reshape,
    but its input is made contiguous first, which is what its lowering wants and
    what the eager result is anyway.
    """
    touched = 0
    seen = set()
    pending = list(holders)
    while pending:
        holder = pending.pop()
        if holder in seen:
            continue
        seen.add(holder)
        holder_value = _node_meta_value(holder)
        if not isinstance(holder_value, torch.Tensor):
            continue
        for user in list(holder.users):
            if not user.args or user.args[0] is not holder:
                continue
            relation = _reindex_relation(user, holder_value)
            if not relation:
                continue
            value = _node_meta_value(user)
            if relation == "view":
                sizes, strides = _layout_extents(value)
                symbols = _symbol_nodes(gm)
                args = (
                    _layout_args(sizes, symbols, before=user),
                    _layout_args(strides, symbols, before=user),
                )
                with gm.graph.inserting_before(user):
                    replacement = _as_strided_on(
                        gm, alloc, args[0], args[1], 0, user, value)
                user.replace_all_uses_with(replacement)
                gm.graph.erase_node(user)
                pending.append(replacement)
            else:
                with gm.graph.inserting_before(user):
                    copied = gm.graph.call_function(
                        torch.ops.aten.clone.default,
                        args=(holder,),
                        kwargs={"memory_format": torch.contiguous_format},
                    )
                # The copy keeps the buffer's shape and becomes contiguous.
                _stamp(copied, user, _plain_meta(holder_value, holder_value.shape))
                user.replace_input_with(holder, copied)
            touched += 1
    return touched


def normalize_fresh_allocation_layouts_(gm: GraphModule) -> int:
    """Make every strided allocation FXRT can actually serve.

    FXRT only allocates contiguous buffers and rejects an ``empty_strided`` that
    asks for anything else. Inductor does emit permuted output layouts -- a
    fused kernel writing a matmul result straight into the transposed buffer,
    for one -- so the buffer is allocated plain and the layout is spelled out as
    an ``as_strided`` where it is consumed: at the compiled-kernel boundary, and
    folded into every view that follows, because FXRT's View op cannot read a
    non-contiguous input.

    Only dense permutations are rewritten -- symbolic extents included, since a
    permutation covers exactly ``numel()`` elements no matter what the extents
    are -- so the contiguous allocation is always large enough.

    Args:
        gm: The host FX graph, modified in place.

    Returns:
        The number of allocations rewritten.
    """
    rewritten = 0
    symbols: Optional[Dict[str, Node]] = None
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or _target_name(node.target) not in _STRIDED_ALLOC_NAMES:
            continue
        value = _node_meta_value(node)
        if not isinstance(value, torch.Tensor):
            continue
        if _contiguous_layout(value):
            # The buffer is already addressable, but inductor spells its layout
            # out with ``empty_strided``. FXRT's converter only accepts strides
            # it can prove contiguous at conversion time, which a stride that
            # still carries a symbol is not, so such a buffer is allocated
            # without spelling its strides out at all. Strides that only differ
            # in a length-1 dimension never reach the address arithmetic and are
            # simply restated.
            if len(node.args) < 2:
                continue
            contiguous = _contiguous_strides(list(value.shape))
            if not all(isinstance(extent, int) for extent in contiguous):
                if not _is_strided_empty(node.target):
                    continue
                node.target = torch.empty
                node.args = (node.args[0],) + tuple(node.args[2:])
                rewritten += 1
                continue
            if _strides_are_canonical(node.args[1], contiguous):
                continue
            if symbols is None:
                symbols = _symbol_nodes(gm)
            node.args = (
                node.args[0], _layout_args(contiguous, symbols, before=node)
            ) + tuple(node.args[2:])
            rewritten += 1
            continue
        if not _is_dense_permutation(value) or len(node.args) < 2:
            continue
        sizes_arg, strides_arg = node.args[0], node.args[1]
        sizes = list(value.shape)
        contiguous = _contiguous_strides(sizes)
        plain_value = value.as_strided(sizes, contiguous, 0)
        view_value = plain_value.as_strided(sizes, list(value.stride()), 0)
        if symbols is None:
            symbols = _symbol_nodes(gm)
        with gm.graph.inserting_before(node):
            alloc = gm.graph.call_function(
                node.target,
                args=(sizes_arg, _layout_args(contiguous, symbols, before=node)) + tuple(node.args[2:]),
                kwargs=node.kwargs,
            )
        _stamp(alloc, node, plain_value)
        with gm.graph.inserting_before(node):
            view = _as_strided_on(gm, alloc, sizes_arg, strides_arg, 0, node, view_value)
        # Consumers first, then whatever is left of the buffer node becomes the
        # explicit layout view the compiled kernels are handed.
        _align_views_with_alloc(gm, alloc, [node])
        node.replace_all_uses_with(view)
        gm.graph.erase_node(node)
        rewritten += 1

    if rewritten:
        gm.graph.lint()
        gm.recompile()
    return rewritten


def _replace_uses_except(arg: Node, replacement: Node) -> None:
    """Point every user of `arg` but the inserted view itself at `replacement`.

    ``Node.replace_all_uses_with`` would also rewrite the view that roots at
    `arg`, making the node its own argument.
    """
    for user in list(arg.users):
        if user is replacement:
            continue
        user.replace_input_with(arg, replacement)


def _view_chain_base(arg: Node, value: torch.Tensor) -> Tuple[Node, List[Node]]:
    """Walk `arg` back to the buffer its view chain roots at.

    Returns the base node and the pure view nodes stepped over, nearest first.
    """
    base = arg
    stepped: List[Node] = []
    while base.op == "call_function" and base.target in _PURE_VIEW_TARGETS:
        if not base.args:
            break
        parent = base.args[0]
        if not isinstance(parent, Node):
            break
        if not _shares_storage(_node_meta_value(parent), value):
            break
        stepped.append(base)
        base = parent
    return base, stepped


def _symbol_nodes(gm: GraphModule) -> Dict[str, Node]:
    """Map every symbolic extent in the graph to the node that produces it."""
    symbols: Dict[str, Node] = {}
    for node in gm.graph.nodes:
        value = _node_meta_value(node)
        if isinstance(value, torch.SymInt):
            symbols.setdefault(str(value.node.expr), node)
    return symbols


def _as_graph_extent(value, symbols: Dict[str, Node], what: str, arg: Node):
    """Turn a size / stride / offset into something an FX node can take."""
    if isinstance(value, int):
        return value
    if isinstance(value, Node):
        # Already a graph node computing this extent (e.g. a shared ``32*s64``
        # symint node reused as a stride); use it as-is.
        return value
    if isinstance(value, torch.SymInt):
        expr = value.node.expr
        if expr.is_Integer:
            return int(expr)
        node = symbols.get(str(expr))
        if node is None:
            raise RuntimeError(
                f"Cannot pin down the layout of compiled-kernel arg '{arg.name}': its "
                f"{what} is the symbolic extent '{expr}', which no node in the host "
                f"graph produces, so it cannot be spelled out as an as_strided argument."
            )
        return node
    raise RuntimeError(
        f"Cannot pin down the layout of compiled-kernel arg '{arg.name}': its {what} "
        f"is {value!r}, expected an int or SymInt."
    )


_ALLOC_TARGETS = (
    torch.ops.aten.empty_strided.default,
    torch.ops.aten.empty_strided,
    torch.empty_strided,
)


def _is_dense_direct_alloc(node: Node, value: torch.Tensor) -> bool:
    """Whether `node` directly allocates a dense-but-non-contiguous buffer.

    FXRT allocates contiguous, so an ``empty_strided`` whose strides are not
    contiguous cannot be lowered as-is. When the layout is still dense and
    non-overlapping, it addresses exactly as many elements as a contiguous
    buffer of the same shape, so it can be reproduced as a contiguous
    allocation viewed with those strides (see :func:`_synthesize_contiguous_base`).
    A layout that is not dense (gaps, overlaps) has no such equivalent and is
    left for the caller to reject.
    """
    if node.op != "call_function" or node.target not in _ALLOC_TARGETS:
        return False
    from torch._prims_common import (  # pylint: disable=import-outside-toplevel
        is_non_overlapping_and_dense_or_false,
    )
    return bool(is_non_overlapping_and_dense_or_false(value))


def _synthesize_contiguous_base(
        gm: GraphModule, arg: Node, value: torch.Tensor, symbols: Dict[str, Node]
) -> Tuple[Node, torch.Tensor]:
    """Insert a contiguous ``empty_strided`` of `value`'s shape to root a view on.

    The allocation mirrors the original node's ``aten.empty_strided`` call -- same
    op and same kwargs (notably ``device``, whose spelling FXRT's device
    resolution is sensitive to: the inductor nodes carry ``device='npu'`` as a
    bare string, and reconstructing a ``torch.device('npu', 0)`` instead makes
    FXRT fall back to CPU) -- only with contiguous strides. ``_as_graph_extent``
    turns the (possibly symbolic) sizes and the contiguous strides derived from
    them into graph-visible extents. The returned fake lets the caller express
    the view as a genuine as_strided of this base, keeping storage identity.
    """
    sizes = [_as_graph_extent(dim, symbols, "size", arg) for dim in value.shape]
    base_fake = value.new_empty(value.shape)  # contiguous: offset 0, dense strides
    cstrides = [_as_graph_extent(st, symbols, "stride", arg) for st in base_fake.stride()]
    kwargs = dict(arg.kwargs) if arg.op == "call_function" else {}
    kwargs.setdefault("dtype", value.dtype)
    kwargs.setdefault("device", value.device)
    with gm.graph.inserting_after(arg):
        base = gm.graph.call_function(
            torch.ops.aten.empty_strided.default, args=(sizes, cstrides), kwargs=kwargs,
        )
    base.meta["val"] = base_fake
    base.meta["example_value"] = base_fake
    return base, base_fake


def _rewrite_arg_to_as_strided(
        gm: GraphModule, arg: Node, value: torch.Tensor, symbols: Dict[str, Node],
        *, preserve_feed_offsets: bool = False,
) -> Node:
    """Insert an as_strided that reproduces `value`'s layout on a base buffer.

    When `arg` is a view of another buffer, the as_strided roots at that buffer.
    When `arg` is produced directly by a dense ``empty_strided`` allocation,
    there is no such buffer, so a contiguous one of the same extent is
    synthesized and the layout expressed as a view over it. A direct allocation
    whose layout is not dense cannot be reproduced this way and is rejected.
    """
    base, _ = _view_chain_base(arg, value)
    anchor = arg
    synthesized_base_fake = None
    self_rooted = base is arg
    # A placeholder is not produced by FXRT: it arrives from the caller with its
    # true (guarded) layout, so re-root the as_strided on the placeholder itself
    # (an identity view that redeclares the layout the kernel was built for).
    if self_rooted and arg.op != "placeholder":
        if not _is_dense_direct_alloc(arg, value):
            raise RuntimeError(
                f"Compiled-kernel arg '{arg.name}' needs the layout "
                f"stride={tuple(value.stride())} offset={value.storage_offset()}, but it is "
                f"produced directly by '{arg.target}' rather than by a view of another "
                f"buffer, and its layout is not dense, so FXRT cannot reproduce it on a "
                f"contiguous allocation. The kernel would index memory that is not laid "
                f"out the way it expects."
            )
        base, synthesized_base_fake = _synthesize_contiguous_base(gm, arg, value, symbols)
        anchor = base

    base_value = value if self_rooted else _node_meta_value(base)
    feed_root = base.op == "placeholder"
    if not self_rooted and not (preserve_feed_offsets and feed_root) and (
            not isinstance(base_value, torch.Tensor)
            or not _same_extent(base_value.storage_offset(), 0)):
        raise RuntimeError(
            f"Compiled-kernel arg '{arg.name}' roots at '{base.name}', which does not start "
            f"at the beginning of its storage; as_strided composes storage offsets, so the "
            f"kernel's arg would land at the wrong element."
        )

    sizes = [_as_graph_extent(dim, symbols, "size", arg) for dim in value.shape]
    strides = [_as_graph_extent(dim, symbols, "stride", arg) for dim in value.stride()]
    # Converter feeds use storage-base pointers plus a live offset endpoint.
    # Express an inherited view as its delta from the *sampled feed*, so the
    # current Execute's origin replaces that sample rather than adding to it.
    # Explicit public absolute views terminate this inheritance.
    feed_relative = preserve_feed_offsets and feed_root and not has_absolute_as_strided_offset(arg)
    storage_offset = value.storage_offset()
    if feed_relative:
        if not isinstance(base_value, torch.Tensor):
            raise RuntimeError(f"Compiled-kernel arg '{arg.name}' has no feed layout metadata")
        storage_offset = storage_offset - base_value.storage_offset()
    offset = _as_graph_extent(storage_offset, symbols, "storage offset", arg)

    # Right after the node it roots at, so every later user can move across
    # without breaking topological order.
    with gm.graph.inserting_after(anchor):
        replacement = gm.graph.call_function(torch.as_strided, args=(base, sizes, strides, offset))
    replacement.meta.update(arg.meta)
    set_view_offset_semantics(
        replacement, ViewOffsetSemantics.FEED_RELATIVE if feed_relative else ViewOffsetSemantics.ABSOLUTE
    )
    if synthesized_base_fake is not None:
        # The view must share the synthesized base's storage, not the original
        # allocation's, so a later pass that re-walks this node (e.g. when the
        # buffer feeds a second kernel) can trace it back to the contiguous base.
        view_fake = synthesized_base_fake.as_strided(
            value.shape, value.stride(), value.storage_offset()
        )
        replacement.meta["val"] = view_fake
        replacement.meta["example_value"] = view_fake
    return replacement


def _erase_unused_view_nodes(gm: GraphModule, candidates) -> None:
    """Drop the view nodes a rewrite bypassed, so nothing materializes for nobody."""
    for node in reversed(list(gm.graph.nodes)):
        if node in candidates and not node.users:
            gm.graph.erase_node(node)


def _flat_extent(sizes, strides, symbols, node):
    """Element count a (sizes, strides) layout can address: sum((size-1)*stride)+1.

    A layout's reachable offsets are exactly the multiples-of-stride grid up to this
    count, so a contiguous allocation of this many elements always covers it --
    whatever the strides are, overlapping or not. Sizes and strides may both be
    symbolic (graph extents), where the arithmetic is carried by SymInt.
    """
    def _extent(value, what):
        if isinstance(value, int):
            return value
        return _as_graph_extent(value, symbols, what, node)

    total = 1
    for dim, stride in zip(sizes, strides):
        total = total + (_extent(dim, "size") - 1) * _extent(stride, "stride")
    return total


def _has_symbolic(values) -> bool:
    """Whether an extent list carries a symbolic value (SymInt or a computing node)."""
    return any(isinstance(v, (torch.SymInt, Node)) for v in values)


def _rewrite_noncontiguous_empty_strided_(gm: GraphModule) -> int:
    """Express every symbolic or non-contiguous ``empty_strided`` as alloc + as_strided.

    FXRT's own ``empty_strided`` only allocates contiguous layouts, and the converter
    can only prove contiguity for concrete strides. A symbolic stride such as
    ``32*s64`` *is* contiguous, but the converter compares expression trees rather
    than proving them equal, so it rejects it. Rewriting keeps symbolic
    ``empty_strided`` out of the converter entirely: the strides move onto an
    ``as_strided``, which the converter does lower symbolically.

    Two shapes of rewrite, chosen by what the layout actually is:

    * Dense/contiguous (``_layout_is_plain``) -- allocate ``empty(sizes)``, whose
      contiguous strides are exactly the ones needed, and view it with the original
      strides. No stride arithmetic at all, so symbolic sizes and strides both work.
    * Non-contiguous with concrete sizes and strides -- allocate the flat element
      count the layout addresses (see :func:`_flat_extent`) and view it.

    Anything else (symbolic *and* non-contiguous) is left for the caller to reject:
    its flat extent would need symbolic stride arithmetic the graph cannot express.
    """
    symbols = None
    rewritten = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _ALLOC_TARGETS:
            continue
        value = _node_meta_value(node)
        if not isinstance(value, torch.Tensor):
            continue
        args = node.args
        if len(args) < 2 or not isinstance(args[0], (list, tuple)) or not isinstance(args[1], (list, tuple)):
            continue
        sizes, strides = list(args[0]), list(args[1])
        if not sizes or len(sizes) != len(strides):
            continue
        plain = _layout_is_plain(value)
        symbolic = _has_symbolic(sizes) or _has_symbolic(strides)
        if plain and not _has_symbolic(strides):
            # Concrete contiguous layout: FXRT allocates this natively and the
            # converter proves it. Symbolic sizes are valid when the contiguous strides
            # remain concrete, so keep those allocations native too.
            continue
        if plain:
            alloc_sizes = sizes
        elif not symbolic:
            if symbols is None:
                symbols = _symbol_nodes(gm)
            alloc_sizes = [_flat_extent(sizes, strides, symbols, node)]
        else:
            continue  # symbolic non-contiguous: needs symbolic stride arithmetic.

        kwargs = dict(node.kwargs)
        kwargs.setdefault("dtype", value.dtype)
        kwargs.setdefault("device", value.device)
        # Fake for the base must be contiguous with the allocation's own shape. In the
        # plain case that shape is value's (a symbolic graph extent, not an int tuple),
        # so build it from `value` rather than from the node args.
        base_fake = value.new_empty(value.shape) if plain else value.new_empty(tuple(alloc_sizes))
        with gm.graph.inserting_before(node):
            base = gm.graph.call_function(torch.ops.aten.empty.memory_format, args=(alloc_sizes,), kwargs=kwargs)
            base.meta["val"] = base_fake
            base.meta["example_value"] = base_fake
            view = gm.graph.call_function(
                torch.ops.aten.as_strided.default,
                args=(base, tuple(sizes), tuple(strides), value.storage_offset()),
            )
        view_fake = base_fake.as_strided(value.shape, value.stride(), value.storage_offset())
        view.meta["val"] = view_fake
        view.meta["example_value"] = view_fake
        _mark_absolute_storage_offset(view)
        node.replace_all_uses_with(view)
        gm.graph.erase_node(node)
        rewritten += 1
    if rewritten:
        gm.graph.lint()
        gm.recompile()
    return rewritten
    if rewritten:
        gm.graph.lint()
        gm.recompile()
    return rewritten


def restore_compiled_kernel_arg_layouts_(gm: GraphModule, *, preserve_feed_offsets: bool = False) -> int:
    """Give every compiled kernel its arguments in the layout it was built for.

    A compiled kernel's index arithmetic is frozen against the layout the FX node
    carries, so an argument asking for anything a plain contiguous allocation does
    not provide is rewritten to an explicit as_strided on the buffer its view chain
    roots at. Arguments that a contiguous allocation already addresses correctly are
    left alone -- which is nearly all of them.

    Raises when the layout cannot be reproduced. Handing the kernel a tensor it will
    misaddress is silently wrong, which is far worse than failing here.

    Args:
        gm: The host FX graph, modified in place.
        preserve_feed_offsets: The converter supplies feed storage-base pointers
            and a separate live origin. Preserve that contract at creation;
            the basic runtime retains its existing tensor argument contract.

    Returns:
        The number of arguments rewritten.
    """
    symbols: Optional[Dict[str, Node]] = None
    bypassed = set()
    stale_allocs = []
    created = set()
    rewritten = 0

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or not is_compiled_kernel_wrapper_mutation_target(node.target):
            continue
        hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)

        for index, arg in enumerate(hop_args):
            if not isinstance(arg, Node):
                continue
            # A replacement this pass already synthesized is correct by
            # construction (a view rooted at a contiguous base); re-walking it
            # would needlessly nest another as_strided.
            if arg in created:
                continue
            value = _node_meta_value(arg)
            if not isinstance(value, torch.Tensor) or _layout_is_plain(value):
                continue

            if symbols is None:
                symbols = _symbol_nodes(gm)
            _, stepped = _view_chain_base(arg, value)
            # A dense direct allocation is rewritten into a contiguous base plus
            # a view; the original empty_strided node is then dead and must not
            # survive to be lowered (FXRT's empty_strided rejects its stride).
            direct_alloc = _is_dense_direct_alloc(arg, value)
            replacement = _rewrite_arg_to_as_strided(
                gm, arg, value, symbols, preserve_feed_offsets=preserve_feed_offsets
            )
            created.add(replacement)

            if index in mutated_arg_indices:
                # The lowering rebinds a mutated arg's node to the kernel op so
                # everything downstream reads what the kernel wrote. Move every
                # user across, or the ones left behind would read the buffer as
                # it was before the launch.
                _replace_uses_except(arg, replacement)
            else:
                args = list(node.kwargs.get("args"))
                args[index] = replacement
                node.kwargs = {**node.kwargs, "args": tuple(args)}

            if direct_alloc:
                stale_allocs.append(arg)
            bypassed.update(stepped)
            rewritten += 1

    if rewritten:
        for alloc in dict.fromkeys(stale_allocs):
            if not alloc.users:
                gm.graph.erase_node(alloc)
        _erase_unused_view_nodes(gm, bypassed)
        gm.graph.lint()
        gm.recompile()
    return rewritten


def lower_compiled_kernel_autofuse_node(
        node,
        executor,
        env,
        sym_mgr,
        get_node_meta_value,
        add_tuple_getitem_node,
):
    """Lower an AutoFuse mutation HOP to the native FXRT Autofuse op.

    This helper is called only by :mod:`fxrt.fx_backend`; the backend
    owns the opt-in decision, while this function only implements native
    lowering and leaves generic compiled-kernel/fx_converter behavior intact.
    """
    if not is_compiled_kernel_wrapper_mutation_target(node.target):
        return False
    kernel_idx = get_compiled_kernel_index(node)
    compiled_kernel = resolve_compiled_kernel(kernel_idx)
    metadata_fn = getattr(compiled_kernel, "fxrt_converter_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    if not isinstance(metadata, dict) or metadata.get("backend") != "inductor_autofuse":
        return False

    # The adapter is per kernel and sits beside kernel.so; wrapper.so may be a
    # shared wrapper_common artifact. Use explicit metadata to retain both paths.
    adapter_path = metadata.get("adapter_path")
    kernel_path = metadata.get("kernel_path")
    if not isinstance(adapter_path, str) or not adapter_path:
        raise RuntimeError("AutoFuse kernel metadata requires a non-empty adapter_path")
    if not isinstance(kernel_path, str) or not kernel_path:
        raise RuntimeError("AutoFuse kernel metadata requires a non-empty kernel_path")
    wrapper_path = metadata.get("wrapper_path")
    if not isinstance(wrapper_path, str) or not wrapper_path:
        wrapper_path = os.path.join(os.path.dirname(adapter_path), "wrapper.so")
    kernel_key = metadata.get("kernel_key", "")
    if kernel_key is None:
        kernel_key = ""
    if not isinstance(kernel_key, str):
        raise RuntimeError("AutoFuse kernel metadata kernel_key must be a string")

    # Delay importing and running the codegen until native AutoFuse is enabled.
    from fxrt.autofuse_codegen import (  # pylint: disable=import-outside-toplevel
        generate_wrapper_stub,
        validate_wrapper_arguments,
    )

    try:
        stub_path, wrapper_arg_types = generate_wrapper_stub(wrapper_path)
    except Exception as exc:
        raise RuntimeError(f"AutoFuse wrapper ABI codegen failed for {wrapper_path}: {exc}") from exc

    hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)
    runtime_args = [get_node_meta_value(arg) if isinstance(arg, Node) else arg for arg in hop_args]
    try:
        validate_wrapper_arguments(wrapper_arg_types, runtime_args)
    except Exception as exc:
        raise RuntimeError(f"AutoFuse wrapper ABI argument validation failed: {exc}") from exc
    output_nodes = get_compiled_kernel_mutation_output_nodes(node)
    if not output_nodes:
        raise RuntimeError("AutoFuse compiled kernel requires at least one mutated output buffer")
    output_examples = []
    for output_node in output_nodes:
        example_value = get_node_meta_value(output_node)
        if example_value is None:
            raise RuntimeError(
                f"AutoFuse output buffer node '{output_node.name}' is missing example_value/val metadata"
            )
        output_examples.append(example_value)

    # [generated typed shim, wrapper.so, kernel.so, kernel_key, mutated indices, original call args]
    input_nodes = [
        executor.add_value_node(sym_mgr.from_torch_with_sym(stub_path)),
        executor.add_value_node(sym_mgr.from_torch_with_sym(wrapper_path)),
        executor.add_value_node(sym_mgr.from_torch_with_sym(kernel_path)),
        executor.add_value_node(sym_mgr.from_torch_with_sym(kernel_key)),
        executor.add_value_node(sym_mgr.from_torch_with_sym(tuple(mutated_arg_indices))),
    ]
    input_nodes += map_hop_args(hop_args, env, executor, sym_mgr)

    if len(output_examples) == 1:
        output_value = sym_mgr.from_torch_with_sym(output_examples[0])
        kernel_node = executor.add_op_node(Op.autofuse_call, input_nodes, output_value)
        env[node] = kernel_node
        env[output_nodes[0]] = kernel_node
        return True

    tuple_output = sym_mgr.from_torch_with_sym(tuple(output_examples))
    tuple_node = executor.add_op_node(Op.autofuse_call, input_nodes, tuple_output)
    env[node] = tuple_node
    for idx, (output_node, output_example) in enumerate(zip(output_nodes, output_examples)):
        output_value = sym_mgr.from_torch_with_sym(output_example)
        env[output_node] = add_tuple_getitem_node(executor, sym_mgr, tuple_node, idx, output_value)
    return True


_VIEW_TARGETS = tuple(
    target
    for target in (
        getattr(torch.ops.aten.view, "default", None),
        getattr(torch.ops.aten.reshape, "default", None),
        getattr(torch.ops.aten._unsafe_view, "default", None),
        torch.reshape,
    )
    if target is not None
)


def _rewrite_noncontiguous_view_(gm: GraphModule) -> int:
    """Materialize non-contiguous views that FXRT cannot represent directly.

    Keep a reshape unchanged when it can be represented as a view. Its original
    strides may be consumed by later as_strided nodes; cloning first would make
    those strides index a different layout.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _VIEW_TARGETS:
            continue
        if not node.args:
            continue
        src = node.args[0]
        if not isinstance(src, Node):
            continue
        value = _node_meta_value(src)
        if not isinstance(value, torch.Tensor) or value.is_contiguous():
            continue
        if node.target is torch.ops.aten.reshape.default:
            try:
                value.view(node.args[1])
            except Exception:
                pass
            else:
                continue
        with gm.graph.inserting_before(node):
            cont = gm.graph.call_function(
                torch.ops.aten.clone.default, (src,), {"memory_format": torch.contiguous_format}
            )
        cval = value.clone(memory_format=torch.contiguous_format)
        cont.meta["val"] = cval
        cont.meta["example_value"] = cval
        node.args = (cont,) + tuple(node.args[1:])
        try:
            shape = node.args[1]
            if node.target in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
                nval = cval.view(shape)
            else:
                nval = cval.reshape(shape)
            node.meta["val"] = nval
            node.meta["example_value"] = nval
        except Exception:
            pass
        changed += 1
    if changed:
        gm.graph.lint()
        gm.recompile()
    return changed
