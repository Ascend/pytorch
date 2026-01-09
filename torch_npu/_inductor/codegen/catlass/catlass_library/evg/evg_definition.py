from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, List, Set, Union

import networkx as nx
from sympy import Symbol

from ..library import BroadcastType, DataType, DataTypeTag, EpilogueOpVectorToScalar
from .node import (CastNode, ComputeNode, ConstantNode, LoadNode, NodeBase,
                   StoreNode, TopoVisitorNode)
from .node_impl import NoOpImpl

if TYPE_CHECKING:
    from ..evg_extension import EpilogueVisitorGraph


# ============ Graph Passes ============
def EliminateDupAndDeadNodes(dag: EpilogueVisitorGraph):
    """
    1) Eliminate duplicated nodes:
        - Two nodes are considered duplicates if:
            * they are the same node kind (ComputeNode, CastNode, ...)
            * they have the same operator identity (for ComputeNode: `fn`;
                for CastNode: from/to element types, etc.)
            * they have identical ordered inputs (same input nodes in the same edge positions)
        - On duplicate detection, we keep the first seen node as canonical and replace
            all outgoing edges from the duplicate node to point from the canonical node.
            Then we remove the duplicate node.

    2) Remove dead nodes:
        - Compute all nodes that are ancestors (via predecessors) of the output store node(s).
        - Any node not in that ancestor set is dead and will be removed.

    This modifies the graph in-place via the existing `add_node`, `remove_node`,
    `add_edge`, `remove_edge`, `get_sorted_inputs`, `get_outputs`, and similar helpers.
    """
    # Helper to build a canonical signature for a node.
    def node_signature(node):
        # Gather ordered input ids (by edge pos)
        try:
            inputs = dag.get_sorted_inputs(node)
        except Exception:
            inputs = []
        input_ids = tuple(inp.name for inp in inputs)

        if isinstance(node, ComputeNode):
            # include the op identity (fn) and possibly element type
            op_id = getattr(node, "fn", None)
            return ("ComputeNode", op_id, input_ids)
        if isinstance(node, CastNode):
            # include cast types
            from_e = getattr(node, "from_element", None)
            to_e = getattr(node, "to_element", None)
            return ("CastNode", from_e, to_e, input_ids)
        if isinstance(node, LoadNode):
            # Loads are unique by name (and marked inputs) — normally we do not dedupe loads.
            return ("LoadNode", node.name)
        if isinstance(node, StoreNode):
            # Stores are unique by name
            return ("StoreNode", node.name)
        # Fallback: use op name and inputs
        return (type(node).__name__, node.metadata.op, input_ids)

    # 1) Remove duplicated nodes
    signature_map: dict = {}
    # Work on a snapshot of nodes in topological order to avoid concurrent modification issues.
    topo_nodes = list(dag.topological_nodes())

    for node in topo_nodes:
        # Node may have been removed during previous iterations
        if not dag.has_node(node):
            continue

        # Remove the StoreNode after the ConstantNode
        if isinstance(node, StoreNode):
            input_node = dag.get_inputs(node)[0]
            if isinstance(input_node, ConstantNode):
                output_node = dag.get_outputs(node)[0]
                dag.add_edge(input_node, output_node)
                dag.remove_edge(node, output_node)
                dag.remove_node(node)
                continue

        # Skip store nodes and loads for dedup unless you want to dedupe them too.
        if isinstance(node, (LoadNode, StoreNode)):
            continue

        sig = node_signature(node)
        # If signature is seen and canonical node exists, replace usages
        if sig in signature_map:
            canonical = signature_map[sig]
            if canonical is node:
                continue

            # Rewire all outputs of `node` to `canonical` with preserving pos
            outputs = list(dag.get_outputs(node))
            for out_node in outputs:
                pos = dag.get_edge_pos(node, out_node)
                # Add new edge canonical -> out_node (if not already present with same pos)
                # Guard against creating duplicate edges of same pos
                already_has = False
                for pred in dag.get_inputs(out_node):
                    if pred == canonical:
                        # If the existing edge has same pos, consider it already wired.
                        if dag.get_edge_pos(canonical, out_node) == pos:
                            already_has = True
                            break
                if not already_has:
                    dag.add_edge(canonical, out_node, pos=pos)

                dag.remove_edge(node, out_node)
                dag.remove_node(node)
        else:
            # First time seeing this signature - register it
            signature_map[sig] = node

    # 2) Remove dead nodes
    # Find output store nodes.
    output_nodes = [
        n
        for n in dag.topological_nodes()
        if isinstance(n, StoreNode) and n.metadata.is_output
    ]
    if not output_nodes:
        raise RuntimeError("The parsed DAG has no output store node.")

    reachable = set()
    # We'll perform reverse DFS from each output to collect all ancestors
    graph_rev = dag._graph.reverse(copy=False)
    for out in output_nodes:
        for n in nx.dfs_preorder_nodes(graph_rev, source=out):
            reachable.add(n)

    # Any node not in reachable is dead -> remove
    all_nodes = list(dag.topological_nodes())
    dead_nodes = [n for n in all_nodes if n not in reachable]
    # Remove dead nodes except maybe LoadNodes that you want to keep (we remove them as dead too)
    for dn in dead_nodes:
        # skip outputs (should not be in dead_nodes) and skip canonical check
        if isinstance(dn, StoreNode) and getattr(dn.metadata, "is_output", False):
            continue
        if not dag.has_node(dn):
            continue
        dag.remove_node(dn)


def ScalarOpsIdentification(dag: EpilogueVisitorGraph):
    """
    Identity Scalar Mulitplication and Add
    """
    all_nodes = dag.topological_nodes()
    for node in all_nodes:
        if isinstance(node, ComputeNode) and node.fn in EpilogueOpVectorToScalar:
            input_nodes = dag.get_inputs(node)
            for input_node in input_nodes:
                if isinstance(input_node, ConstantNode):
                    node.fn = EpilogueOpVectorToScalar[node.fn]
                    node._scalar_values[f"{node.name}_scalar_{len(node._scalar_values)}"] = (input_node.value, input_node.metadata.element)
                    dag.remove_edge(input_node, node)
                    dag.remove_node(input_node)


def InferShape(dag: EpilogueVisitorGraph):
    """
    Infer each node's output shape from its input nodes.
    - Uses lexicographical topological order so inputs are inferred before users.
    - Broadcasts shapes using left-padding with 1s, then matching dimensions from the end.
    """
    def pad_left(shape_list, target_len):
        # returns a new list padded on the left with 1s to target_len
        if shape_list is None:
            return [1] * target_len
        if len(shape_list) >= target_len:
            return list(shape_list)
        return [1] * (target_len - len(shape_list)) + list(shape_list)

    # Walk nodes in topological order so inputs are available
    for node in dag.topological_nodes():
        # Get ordered inputs (by edge pos)
        inputs = dag.get_sorted_inputs(node)

        # No inputs: for many nodes this is a scalar or pre-filled metadata; leave as-is
        if not inputs:
            continue

        # Collect source shapes
        src_shapes = []
        for src in inputs:
            meta = src.metadata
            src_shapes.append(meta.shape)

        # Infer broadcasted shape
        shape = None
        for src_shape in src_shapes:
            if shape is None:
                # start with first shape
                shape = list(src_shape)
                continue

            # pad shapes to same length on the left with 1s
            max_len = max(len(shape), len(src_shape))
            a = pad_left(shape, max_len)
            b = pad_left(list(src_shape), max_len)

            # build result from right-to-left
            result_rev = []
            for dim_a, dim_b in zip(reversed(a), reversed(b)):
                if dim_a == 1:
                    result_rev.append(dim_b)
                elif dim_b == 1:
                    result_rev.append(dim_a)
                elif dim_a == dim_b:
                    result_rev.append(dim_a)
                else:
                    # construct helpful error message listing input shapes
                    shapes_msg = ", ".join(f"{inp.name}{tuple(inp.metadata.shape)}" for inp in inputs)
                    raise RuntimeError(f"Dimension mismatch between {shapes_msg}.")
            # reverse result_rev to get normal order
            shape = list(reversed(result_rev))

        final_shape = tuple(shape) if shape else ()

        node.metadata.shape = final_shape


def BroadcastPropagation(dag: EpilogueVisitorGraph):
    """
    Decide if any node needs broadcast and whether row- or col-broadcast is required.
    """

    def check_broadcast(src_shape, dst_shape):
        """
        Check if src_shape can broadcast to dst_shape according to Numpy's broadcast rule.

        Returns (can_broadcast: bool, row_broadcast: bool, col_broadcast: bool)

        - We align shapes on the right and compare dimensions from right to left.
        - A dimension is compatible if either equal or one of them is 1.
        - If src dim == 1 and dst dim != 1, src is being broadcast along that dst axis.
        - If that dst axis is the last axis (rightmost), treat it as a column broadcast.
        - Otherwise treat it as a row broadcast.
        - If dst dim == 1 and src dim != 1 (including when dst lacks the axis), src cannot be broadcast
        to dst (return False).
        """
        a = list(src_shape)
        b = list(dst_shape)

        i = len(a) - 1
        j = len(b) - 1
        row_broadcast = False
        col_broadcast = False

        # Compare from right to left (align shapes to the right)
        while i >= 0 or j >= 0:
            dim_a = a[i] if i >= 0 else 1
            dim_b = b[j] if j >= 0 else 1

            if dim_a == dim_b:
                # compatible, nothing to record
                pass
            elif dim_a == 1 and dim_b != 1:
                # src is broadcast along this dst axis
                # if this is the last (rightmost) dst axis -> column broadcast,
                # otherwise -> row broadcast
                if j == len(b) - 1:
                    col_broadcast = True
                else:
                    row_broadcast = True
            else:
                return False, row_broadcast, col_broadcast

            i -= 1
            j -= 1

        return True, row_broadcast, col_broadcast

    for node in dag.topological_nodes():
        if isinstance(node, StoreNode):
            continue
        outputs = dag.get_outputs(node)
        if not outputs:
            continue

        out_shapes = [out.metadata.shape for out in outputs]
        # All output shapes must be equal
        first_shape = out_shapes[0]
        if any(s != first_shape for s in out_shapes):
            raise RuntimeError(
                f"Node '{node.name}' has outputs with differing shapes: {out_shapes}"
            )

        src_shape = node.metadata.shape
        dst_shape = first_shape
        ok, row_broadcast, col_broadcast = check_broadcast(src_shape, dst_shape)
        if not ok:
            raise RuntimeError(
                f"Node '{node.name}' shape {src_shape} cannot be broadcast to output shape {dst_shape}"
            )
        if col_broadcast:
            raise RuntimeError(
                "EVG does not support ColumnBroadcast yet!"
            )
        if row_broadcast:
            node.metadata.broadcast = BroadcastType.RowBroadcast
            if len(node.metadata.shape) == 1:
                node.metadata.shape = (1, node.metadata.shape[0])


def DynamicShapeTransfer(dag: EpilogueVisitorGraph):
    """
    Subtitutes all shape infos into a experssion of m & n, which aims to support dynamic shape.
    """
    for node in dag.topological_nodes():
        # Within dynamic shape mode, the symbolic shape shoule be sympy.Symbol type
        if all(not isinstance(shape_i, Symbol) for shape_i in node.metadata.shape):
            continue
        symbolic_shape = []
        for shape_i in node.metadata.shape:
            shape_i = str(shape_i)
            for key, item in dag.symbol_shape_substitution_dict.items():
                shape_i = str(shape_i).replace(key, item)
            symbolic_shape.append(shape_i)
        node.metadata.shape = tuple(symbolic_shape) if symbolic_shape else ()


def SetNodeImpl(dag: EpilogueVisitorGraph):
    """
    Map each node of the EVG to the underlying node impl.
    """
    for node in dag.topological_nodes():
        if isinstance(node, TopoVisitorNode):
            # TopoVisitorNode's impl is already set at its initialization
            continue
        node.get_impl()

    # Eliminate node with NoOpImpl
    for node in dag.topological_nodes():
        if isinstance(node.impl, NoOpImpl):
            input_nodes = dag.get_inputs(node)
            assert len(input_nodes) == 1
            in_node = input_nodes[0]
            for out_node in dag.get_outputs(node):
                pos = dag.get_edge_pos(node, out_node)
                dag.add_edge(in_node, out_node, pos)
                dag.remove_edge(node, out_node)
            dag.remove_node(node)
            

def DAG2Tree(dag: EpilogueVisitorGraph):
    """
    Transform a DAG to Tree by fusing subgraphs containing nodes with multiple outputs.
    """
    def _find_lca(node: NodeBase):
        output_nodes = list(dag.get_outputs(node))
        if not output_nodes:
            return None, None

        reachable_nodes = []
        for s_node in output_nodes:
            reachable_nodes.append(set(dag.all_reachable_nodes(s_node)))
        common_nodes = set.intersection(*reachable_nodes)
        if not common_nodes:
            return None, None

        topo_nodes = dag.topological_nodes()
        lca = min(common_nodes, key=lambda node: topo_nodes.index(node))
        nodes_to_fuse = set.union(*reachable_nodes).difference(common_nodes)
        nodes_to_fuse.add(lca)
        return lca, nodes_to_fuse

    def _fuse_subgraph(nodes_to_fuse: Set[NodeBase], lca: NodeBase):
        """
        Fuse the nodes between node and lca into a single TopoVisitorNode.
        """
        from ..evg_extension import EpilogueVisitorGraph

        # Get all the immediate inputs & outputs of the nodes_to_fuse
        all_input_nodes = set()
        all_output_nodes = set()
        for node in nodes_to_fuse:
            all_input_nodes.update(dag.get_inputs(node))
            all_output_nodes.update(dag.get_outputs(node))
        new_subgraph_nodes = set.union(nodes_to_fuse, all_input_nodes, all_output_nodes)

        subgraph_ = dag._graph.subgraph(new_subgraph_nodes)
        subgraph = EpilogueVisitorGraph()
        for node in subgraph_.nodes:
            new_node = deepcopy(node)
            if node not in nodes_to_fuse:
                new_node.disabled = True
            subgraph.add_node(new_node)
        for edge in subgraph_.edges:
            subgraph.add_edge(
                edge[0].name, edge[1].name, dag.get_edge_pos(edge[0], edge[1])
            )

        tv_node = TopoVisitorNode(
            name=f"tv_{lca.name}",
            metadata=lca.metadata,
            subgraph=subgraph,
            output_node=lca,
        )
        dag.add_node(tv_node)

        # Add input edges
        for idx, node in enumerate(all_input_nodes):
            dag.add_edge(node, tv_node, pos=idx)

        # Replace all uses of lca with TopoVisitorNode
        for node in dag.get_outputs(lca):
            pos = dag.get_edge_pos(lca, node)
            dag.add_edge(tv_node, node, pos)
            dag.remove_edge(lca, node)
        dag.remove_node(lca)

        # Replace all fused nodes
        nodes_to_fuse.remove(lca)
        for node in nodes_to_fuse:
            dag.remove_node(node)

    multiple_output_nodes = [node for node in dag.topological_nodes() if dag.out_degree(node) > 1]

    for node in multiple_output_nodes:
        if not (dag.has_node(node) and dag.out_degree(node) > 1):
            continue

        # Find LCA, nodes_to_fuse
        lca, nodes_to_fuse = _find_lca(node)

        if not lca:
            raise NotImplementedError("No LCA found.")

        _fuse_subgraph(nodes_to_fuse, lca)


class EVGDef:
    def __init__(self, dag: EpilogueVisitorGraph):
        self.dag = dag
        EliminateDupAndDeadNodes(self.dag)
        ScalarOpsIdentification(self.dag)
        InferShape(self.dag)
        BroadcastPropagation(self.dag)
        DynamicShapeTransfer(self.dag)
        SetNodeImpl(self.dag)
        DAG2Tree(self.dag)

    def get_visitor_name(self, node: NodeBase) -> str:
        if not isinstance(node, TopoVisitorNode) and self.dag.in_degree(node) > 0:
            return f"EVG{node.type_name}"
        return node.type_name

    def definition(self):
        nodes = self.dag.topological_nodes()
        evg_str = ""
        # Define 1. individual node type decl
        #        2. epilogue tree node
        #        3. topovisitor node
        for node in nodes:
            if not node.disabled:
                evg_str += self.def_node(node)
            if isinstance(node, TopoVisitorNode):
                evg_str += self.def_subgraph_node(node)
            else:
                # Tree visitor node
                evg_str += self.def_tree_node(node)

        callback_name = self.get_visitor_name(nodes[-1])
        return evg_str, callback_name

    def def_node(self, node: NodeBase):
        if isinstance(node, TopoVisitorNode):
            node_str = ""
            for inner_node in node.subgraph.topological_nodes():
                if not inner_node.disabled:
                    node_str += self.def_node(inner_node)
            return node_str
        return node.impl.type_decl

    def def_tree_node(self, node: NodeBase):
        if self.dag.in_degree(node) == 0:
            return ""

        sorted_input_nodes = self.dag.get_sorted_inputs(node)
        inputs_str_list = [f"    {self.get_visitor_name(i_node)}" for i_node in sorted_input_nodes]
        inputs_str = ",\n".join(inputs_str_list)
        tree_node_str = f"""
using EVG{node.type_name} = Catlass::Epilogue::Fusion::TreeVisitor<
    {node.type_name},
{inputs_str}
>;
"""
        return tree_node_str

    def def_subgraph_node(self, node: NodeBase):
        subgraph = node.subgraph
        subgraph_nodes = subgraph.topological_nodes()

        # define the edge tuple
        edges_str = "tla::tuple<\n"
        for snode in subgraph_nodes[:-1]:
            sorted_input_nodes = subgraph.get_sorted_inputs(snode)
            sorted_input_ids = [str(subgraph_nodes.index(i_node)) for i_node in sorted_input_nodes]
            edge_str = "        tla::seq<" + ", ".join(sorted_input_ids) + ">,\n"
            edges_str += edge_str
        parts = edges_str.rsplit(",", 1)
        edges_str = parts[0] + "\n"
        edges_str += "    >"

        # define the nodes list
        tv_nodes_str_list = []
        for snode in subgraph_nodes[:-1]:
            if snode.disabled:
                tv_nodes_str_list.append(f"    {self.get_visitor_name(snode)}")
            else:
                tv_nodes_str_list.append(f"    {snode.type_name}")

        tv_nodes_str = ",\n".join(tv_nodes_str_list)

        subgraph_str = f"""
using {node.type_name} = Catlass::Epilogue::Fusion::TopologicalVisitor<
    {edges_str},
{tv_nodes_str}
>;
"""
        return subgraph_str


class EVGArg:
    def __init__(self, dag: EpilogueVisitorGraph):
        self.dag = dag

    def generate_graph_args(self) -> str:
        """
        Return Arg Infos and ArgRename for the EVG
        """
        # find output nodes
        output_nodes = [
            node
            for node in self.dag.topological_nodes()
            if self.dag.out_degree(node) == 0
        ]
        output_nodes = [node for node in output_nodes if isinstance(node, StoreNode)]

        if not output_nodes:
            raise ValueError("Cannot find output node in EVG")

        if len(output_nodes) > 1:
            raise ValueError(
                "Find more than one node in EVG, currently, only support one output"
            )

        output_node = output_nodes[0]

        # generate arguments for the given EVG
        final_args = f"typename EVG{output_node.type_name}::Arguments evg_args"
        final_args += self.generate_node_args(output_node, self.dag)
        final_args = self._replace_last_comma_with_semicolon(final_args)

        return (
            final_args,
            None,
        )

    def generate_compute_length(self) -> str:
        """
        Return the compute length arg for the given DAG, which is used as tiling infos
        """
        nodes_element_dict = {}
        self.dag.get_storage_nodes(nodes_element_dict)
        nodes_length_str = "(" + " + ".join([f"({num} * sizeof({DataTypeTag[element]}))" for element, num in nodes_element_dict.items()]) + ")"
        return f"""
constexpr uint32_t computeLength = 216 * 1024 / {nodes_length_str} / 2 / 32 * 32;
"""


    def generate_node_args(
        self, node: NodeBase, graph: EpilogueVisitorGraph, hierarchical_count=0
    ) -> str:
        """
        Genrate arg infos for the given node

        Args:
            node: input node
            graph: Input EVG graph

        Returns:
            Arg infos
        """
        if isinstance(node, TopoVisitorNode):
            topo_inputs = graph.get_sorted_inputs(node)
            tree_result_list = []
            for node_i in topo_inputs:
                tree_result_list.append(
                    self.generate_node_args(node_i, graph, hierarchical_count + 1)
                )
            topo_result = self.generate_topo_visitor_args(
                node, topo_inputs, hierarchical_count + 1
            )
            for i in range(len(tree_result_list)):
                topo_result = topo_result.replace(
                    f"{topo_inputs[i].name}", tree_result_list[i]
                )
            return topo_result

        return self.generate_regular_node_args(node, graph, hierarchical_count)

    def generate_regular_node_args(
        self, node: NodeBase, graph: EpilogueVisitorGraph, hierarchical_count
    ) -> str:
        """generate tree node args"""
        input_nodes = graph.get_sorted_inputs(node)
        result = ""
        if input_nodes:
            result += self._generate_indent(hierarchical_count) + "{\n"

        for input_node in input_nodes:
            input_arg = self.generate_node_args(
                input_node, graph, hierarchical_count + 1
            )
            result += input_arg
        result += (
            self._generate_indent(hierarchical_count + 1)
            + node.impl.args_decl
            + ",\n"
        )
        if input_nodes:
            result = self._remove_last_comma(result)
            result += self._generate_indent(hierarchical_count) + "},\n"
        return result

    def generate_topo_visitor_args(
        self, node: TopoVisitorNode, topo_inputs, hierarchical_count
    ) -> str:
        """generating the topo graph's arguments"""
        subgraph = node.subgraph
        output_node = node.output_node

        # find all reachable nodes in the topo subgraphs
        reachable_nodes = subgraph.topological_nodes()[:-1]
        result = self._generate_indent(hierarchical_count - 1) + "{\n"
        for node in reachable_nodes:
            if node not in topo_inputs:
                result += (
                    self._generate_indent(hierarchical_count)
                    + node.impl.args_decl
                    + ",\n"
                )
            else:
                result += f"{node.name}"
        result = self._remove_last_comma(result)
        result += self._generate_indent(hierarchical_count - 1) + "},\n"

        return result

    def _replace_last_comma_with_semicolon(self, s):
        parts = s.rsplit(",", 1)
        if len(parts) == 1:
            return s

        return ";".join(parts)

    def _remove_last_comma(self, s):
        parts = s.rsplit(",", 1)
        if len(parts) == 1:
            return s

        return parts[0] + parts[1]

    def _generate_indent(self, hierarchical_count: int) -> str:
        return " " * hierarchical_count * 4
