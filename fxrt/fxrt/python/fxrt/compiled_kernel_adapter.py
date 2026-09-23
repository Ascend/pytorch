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
"""

import importlib
from typing import List, Tuple

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
