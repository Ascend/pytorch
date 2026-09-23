"""
FX graph pass: eliminate redundant copy_ when destination is not graph input/output.
"""
from typing import Any

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.immutable_collections import immutable_list
from torch.fx.node import Node


def _replace_node_in_args(val: Any, old_node: Node, new_node: Node) -> Any:
    """
    Recursively replace old_node with new_node in args/kwargs (tuple, list, dict).

    Args:
        val: The value to search and replace within (can be nested structures).
        old_node: The node to be replaced.
        new_node: The replacement node.

    Returns:
        The value with all occurrences of old_node replaced by new_node.
    """
    if val is old_node:
        return new_node
    if isinstance(val, tuple):
        return tuple(_replace_node_in_args(v, old_node, new_node) for v in val)
    if isinstance(val, (list, immutable_list)):
        return tuple(_replace_node_in_args(v, old_node, new_node) for v in val)
    if isinstance(val, dict):
        return {k: _replace_node_in_args(v, old_node, new_node) for k, v in val.items()}
    return val


def _collect_output_nodes(arg, output_nodes):
    """
    Recursively collect output nodes from graph output argument.

    Args:
        arg: The argument to collect nodes from.
        output_nodes: Set to populate with found nodes.
    """
    if isinstance(arg, Node):
        output_nodes.add(arg)
    elif isinstance(arg, (tuple, list)):
        for a in arg:
            _collect_output_nodes(a, output_nodes)
    elif isinstance(arg, dict):
        for v in arg.values():
            _collect_output_nodes(v, output_nodes)


def _get_graph_output_nodes(gm: GraphModule) -> set:
    """
    Return set of nodes that are part of graph output (return value).

    Args:
        gm: The GraphModule to analyze.

    Returns:
        set: Set of output nodes in graph.
    """
    output_nodes = set()
    for node in gm.graph.nodes:
        if node.op != "output":
            continue
        _collect_output_nodes(node.args[0], output_nodes)
        break
    return output_nodes


def _is_copy_node(n: Node) -> bool:
    """
    Check if node is a copy operation (copy_ method or aten.copy_.default).

    Args:
        n: The FX node to check.

    Returns:
        bool: True if node is a copy operation, False otherwise.
    """
    if n.op == "call_method" and n.target == "copy_":
        return True
    if n.op == "call_function":
        aten_copy_default = getattr(torch.ops.aten, "copy_.default", None)
        if aten_copy_default is not None and n.target is aten_copy_default:
            return True
    return False


def _is_linear_node(n: Node) -> bool:
    """
    Check if node is a linear operation.

    Args:
        n: The FX node to check.

    Returns:
        bool: True if node is a linear operation, False otherwise.
    """
    if n.op != "call_function":
        return False

    # pylint: disable=protected-access
    linear_targets = {
        getattr(torch._C._nn, "linear", None),
        getattr(torch.nn.functional, "linear", None),
        getattr(getattr(torch._C, "nn", None), "linear", None),
    }
    aten_linear_default = getattr(torch.ops.aten, "linear", None)
    if aten_linear_default is not None:
        linear_targets.add(getattr(aten_linear_default, "default", None))
    linear_targets.discard(None)

    return n.target in linear_targets


def eliminate_redundant_copy_(gm: GraphModule) -> None:
    """
    Eliminate copy_ when the destination x is a linear result and is neither
    graph input nor output.

    For every user of x that is topologically after the copy_ node, replace
    that use of x with y (the source), then remove the copy_ node.

    Args:
        gm: The GraphModule to optimize.
    """
    graph = gm.graph
    nodes_ordered = list(graph.nodes)
    node_to_index = {n: i for i, n in enumerate(nodes_ordered)}
    output_nodes = _get_graph_output_nodes(gm)

    copy_nodes = [n for n in nodes_ordered if _is_copy_node(n)]

    for copy_node in copy_nodes:
        if len(copy_node.args) < 2:
            continue
        x_node = copy_node.args[0]
        y_node = copy_node.args[1]
        if not isinstance(x_node, Node) or not isinstance(y_node, Node):
            continue
        if x_node.op == "placeholder":
            continue
        if x_node in output_nodes:
            continue
        if not _is_linear_node(x_node):
            continue

        copy_index = node_to_index[copy_node]
        x_users = list(x_node.users)
        for user in x_users:
            if user is copy_node:
                continue
            if node_to_index[user] <= copy_index:
                continue
            new_args = _replace_node_in_args(user.args, x_node, y_node)
            new_kwargs = _replace_node_in_args(user.kwargs, x_node, y_node)
            user.args = new_args
            user.kwargs = new_kwargs

        graph.erase_node(copy_node)
