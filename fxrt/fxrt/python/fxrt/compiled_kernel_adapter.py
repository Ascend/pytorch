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
that layout down -- see :func:`restore_compiled_kernel_arg_layouts_`.
"""

import importlib
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Node

from fxrt.ir import Op

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


def _rewrite_arg_to_as_strided(
        gm: GraphModule, arg: Node, value: torch.Tensor, symbols: Dict[str, Node]
) -> Node:
    """Insert an as_strided that reproduces `value`'s layout on its base buffer."""
    base, _ = _view_chain_base(arg, value)
    if base is arg:
        raise RuntimeError(
            f"Compiled-kernel arg '{arg.name}' needs the layout "
            f"stride={tuple(value.stride())} offset={value.storage_offset()}, but it is "
            f"produced directly by '{arg.target}' rather than by a view of another "
            f"buffer, and FXRT hands a freshly produced tensor to the kernel contiguous. "
            f"The kernel would index memory that is not laid out the way it expects."
        )

    base_value = _node_meta_value(base)
    if not isinstance(base_value, torch.Tensor) or not _same_extent(base_value.storage_offset(), 0):
        raise RuntimeError(
            f"Compiled-kernel arg '{arg.name}' roots at '{base.name}', which does not start "
            f"at the beginning of its storage; as_strided composes storage offsets, so the "
            f"kernel's arg would land at the wrong element."
        )

    sizes = [_as_graph_extent(dim, symbols, "size", arg) for dim in value.shape]
    strides = [_as_graph_extent(dim, symbols, "stride", arg) for dim in value.stride()]
    offset = _as_graph_extent(value.storage_offset(), symbols, "storage offset", arg)

    # Right after the node it replaces, so every user of that node -- which all
    # come later -- can be moved across without breaking topological order.
    with gm.graph.inserting_after(arg):
        replacement = gm.graph.call_function(torch.as_strided, args=(base, sizes, strides, offset))
    # Same tensor, reached a different way: it carries the same metadata.
    replacement.meta.update(arg.meta)
    return replacement


def _erase_unused_view_nodes(gm: GraphModule, candidates) -> None:
    """Drop the view nodes a rewrite bypassed, so nothing materializes for nobody."""
    for node in reversed(list(gm.graph.nodes)):
        if node in candidates and not node.users:
            gm.graph.erase_node(node)


def restore_compiled_kernel_arg_layouts_(gm: GraphModule) -> int:
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

    Returns:
        The number of arguments rewritten.
    """
    symbols: Optional[Dict[str, Node]] = None
    bypassed = set()
    rewritten = 0

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or not is_compiled_kernel_wrapper_mutation_target(node.target):
            continue
        hop_args, mutated_arg_indices = get_compiled_kernel_mutation_args(node)

        for index, arg in enumerate(hop_args):
            if not isinstance(arg, Node):
                continue
            value = _node_meta_value(arg)
            if not isinstance(value, torch.Tensor) or _layout_is_plain(value):
                continue

            if symbols is None:
                symbols = _symbol_nodes(gm)
            _, stepped = _view_chain_base(arg, value)
            replacement = _rewrite_arg_to_as_strided(gm, arg, value, symbols)

            if index in mutated_arg_indices:
                # The lowering rebinds a mutated arg's node to the kernel op so
                # everything downstream reads what the kernel wrote. Move every
                # user across, or the ones left behind would read the buffer as
                # it was before the launch.
                arg.replace_all_uses_with(replacement)
            else:
                args = list(node.kwargs.get("args"))
                args[index] = replacement
                node.kwargs = {**node.kwargs, "args": tuple(args)}

            bypassed.update(stepped)
            rewritten += 1

    if rewritten:
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

    # fx_converter's historical metadata names the adapter shared object and
    # kernel shared object.  The native path intentionally uses only wrapper.so;
    # derive its sibling path without changing the metadata producer.
    adapter_path = metadata.get("adapter_path")
    kernel_path = metadata.get("kernel_path")
    if not isinstance(adapter_path, str) or not adapter_path:
        raise RuntimeError("AutoFuse kernel metadata requires a non-empty adapter_path")
    if not isinstance(kernel_path, str) or not kernel_path:
        raise RuntimeError("AutoFuse kernel metadata requires a non-empty kernel_path")
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
