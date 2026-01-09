"""
Base class for Python EVG fronted
"""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from sympy import Expr, Symbol

from ..catlass_utils import torch_dtype_to_catlass_type
from .evg.evg_definition import EVGArg, EVGDef
from .evg.node import (CastNode, ComputeNode, ConstantNode, LoadNode, NodeBase,
                       NodeMetadata, StoreNode, TopoVisitorNode)
from .library import DataType, EpilogueOp, TileDescription


class EpilogueVisitorGraph:
    """
    Helper class for constructing a DAG from Python EVG function
    """
    def __init__(self):
        self._graph = nx.DiGraph()
        self.nodes_map = {}  # a little fragile
        self.compute_counter = itertools.count()

        # used for dynamic shape, should be reimplemented with more mathematics if permute kind ops are supported
        self.symbol_shape_substitution_dict = {}

    def add_node(self, node: NodeBase) -> None:
        self.check_not_exist(node)
        self._graph.add_node(node)
        self.nodes_map[node.name] = node

    def get_node(self, node_or_name: Union[str, NodeBase]) -> NodeBase:
        if isinstance(node_or_name, str):
            return self.nodes_map[node_or_name]
        return node_or_name

    def add_edge(
        self,
        src_node: Union[str, NodeBase],
        dst_node: Union[str, NodeBase],
        pos: int = 0,
    ):
        self.check_exist(src_node)
        self.check_exist(dst_node)
        src_node = self.get_node(src_node)
        dst_node = self.get_node(dst_node)
        self._graph.add_edge(src_node, dst_node, weight=pos)

    def remove_node(self, node: NodeBase):
        self._graph.remove_node(node)
        del self.nodes_map[node.name]

    def remove_edge(
        self, src_node: Union[str, NodeBase], dst_node: Union[str, NodeBase]
    ):
        src_node = self.get_node(src_node)
        dst_node = self.get_node(dst_node)
        self._graph.remove_edge(src_node, dst_node)

    def has_node(self, node: NodeBase) -> bool:
        return self._graph.has_node(node)

    def check_not_exist(self, node: Union[str, NodeBase]):
        if self.has_node(self.get_node(node)):
            raise SyntaxError(f"Variable '{str(node)}' is already defined before")

    def check_exist(self, node: Union[str, NodeBase]):
        if not self.has_node(self.get_node(node)):
            raise SyntaxError(f"Variable '{str(node)}' is used before definiton")

    def get_storage_nodes(self, nodes_element_dict):
        """
        Returns a dict, containing the infos of the total number of nodes for each type of element
        """
        for node in self.topological_nodes():
            if node.disabled or isinstance(node, StoreNode):
                continue
            if isinstance(node, TopoVisitorNode):
                node.subgraph.get_storage_nodes(nodes_element_dict)
                continue
            element = node.metadata.element
            if element in nodes_element_dict:
                nodes_element_dict[element] += 1
            else:
                nodes_element_dict[element] = 1

    def to_networkx(self) -> nx.DiGraph:
        return self._graph

    def topological_nodes(self) -> List[NodeBase]:
        return list(nx.lexicographical_topological_sort(self._graph))

    def in_degree(self, node: NodeBase) -> int:
        return self._graph.in_degree(node)

    def out_degree(self, node: NodeBase) -> int:
        return self._graph.out_degree(node)

    def get_outputs(self, node: NodeBase) -> List[NodeBase]:
        return list(self._graph.successors(node))

    def get_inputs(self, node: NodeBase) -> List[NodeBase]:
        return list(self._graph.predecessors(node))

    def get_sorted_inputs(self, node: NodeBase) -> List[NodeBase]:
        input_nodes = {self.get_edge_pos(pnode, node): pnode for pnode in self.get_inputs(node)}
        return [input_nodes[key] for key in sorted(input_nodes.keys())]

    def all_reachable_nodes(self, node: NodeBase) -> List[NodeBase]:
        return list(nx.dfs_preorder_nodes(self._graph, source=node))

    def get_edge_pos(self, src_node: NodeBase, dst_node: NodeBase) -> int:
        return self._graph.get_edge_data(src_node, dst_node)["weight"]

    def mark_output(self, name: str, traced_tensor):
        node = self.get_node(name)
        if not isinstance(node, StoreNode):
            raise ValueError(f"Only StoreNode can be marked as output. Got: {name}")
        node.metadata.is_output = True
        node.metadata.op = "auxstore"
        node.metadata.element = traced_tensor.element
        node.metadata.shape = traced_tensor.shape
        node.metadata.stride = traced_tensor.stride

    def add_load_node(self, name: str, traced_tensor):
        if name is None:
            raise ValueError("Node name is not provided")
        if traced_tensor is None:
            raise ValueError(f"Input for {name} is not provided")

        op = "accload" if name == "accum" else "auxload"
        if op == "accload":
            # Within dynamic shape mode, the symbolic shape shoule be sympy.Symbol type
            # Even in dynamic shape mode, the input shape might be (s1, 1) which is (sympy.Symbol, sympy.Int)
            # Obviously, we do not suppose to subtitute all 1s to "n"
            if isinstance(traced_tensor.shape[0], Symbol):
                self.symbol_shape_substitution_dict[str(traced_tensor.shape[0])] = "m"
            if isinstance(traced_tensor.shape[1], Symbol):
                self.symbol_shape_substitution_dict[str(traced_tensor.shape[1])] = "n"
        
        metadata = NodeMetadata(
            op=op,
            shape=traced_tensor.shape,
            element=traced_tensor.element,
        )
        load_node = LoadNode(name, metadata)
        self.add_node(load_node)

    def add_store_node(self, element: DataType, name: str):
        metadata = NodeMetadata(
            op="store",
            element=element,
        )
        node = StoreNode(name, metadata)
        self.add_node(node)

    def add_compute_node(self, op, element: DataType, name=None):
        if name is None:
            name = f"compute_{next(self.compute_counter)}"
        metadata = NodeMetadata(
            op="compute",
            element=element,
        )
        compute_node = ComputeNode(
            name=name,
            metadata=metadata,
            fn=op,
        )
        self.add_node(compute_node)
        return name

    def add_constant_node(self, value, dtype):
        element = DataType(dtype)
        name = f"constant_{value}_{next(self.compute_counter)}"
        metadata = NodeMetadata(
            op="constant",
            element=element,
        )
        constant_node = ConstantNode(
            name=name,
            metadata=metadata,
            value=value,
        )
        self.add_node(constant_node)
        return name

    def add_cast_node(self, dst_type, src_type):
        dst_element = DataType(dst_type)
        src_element = DataType(src_type)
        name = f"cast_{next(self.compute_counter)}"
        metadata = NodeMetadata(
            op="cast",
            element=dst_element,
        )
        cast_node = CastNode(
            name=name,
            metadata=metadata,
            from_element=src_element,
            to_element=dst_element,
        )
        self.add_node(cast_node)
        return name



class PythonEVGParser(EpilogueVisitorGraph, ast.NodeVisitor):
    """
    Transform a Python EVG function to DAG
    """
    def __init__(self):
        """
        e.g. example_inputs = {
            "accum": CatlassTensor(),
            ...
        }
        """
        super().__init__()
        self.identity_inplace_dict: Dict[str, str] = {}
        self.visiting_return = False

    @staticmethod
    def ast_op_bindings(op):
        mapping = {
            ast.Add: EpilogueOp.Add,
            ast.Sub: EpilogueOp.Sub,
            ast.Mult: EpilogueOp.Mul,
            ast.Div: EpilogueOp.Div,
            "relu": EpilogueOp.Relu,
            "max": EpilogueOp.Max,
            "min": EpilogueOp.Min,
            "sigmoid": EpilogueOp.Sigmoid,
            "silu": EpilogueOp.Silu,
        }
        if op not in mapping:
            raise NotImplementedError(f"Operation {op} is not supported")
        return mapping[op]

    def parse(self, fn_src, example_inputs):
        self.example_inputs = example_inputs
        self.source = textwrap.dedent(fn_src)
        self.ast = ast.parse(self.source)
        self.visit(self.ast)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit FunctionDef in ast.NodeVisiter"""
        # processs args
        for arg in node.args.args:
            self.visit(arg)

        # process expression
        for expr in node.body:
            self.visit(expr)

    def visit_arg(self, node: ast.arg):
        arg_name = node.arg
        try:
            input_tensor = self.example_inputs[arg_name]
        except KeyError as e:
            raise RuntimeError(f"Input for {arg_name} is not provided") from e

        self.add_load_node(arg_name, input_tensor)

    def visit_Name(self, node: ast.Name):
        return node.id

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Tuple(self, node: ast.Tuple):
        results = []
        for elt in node.elts:
            results.append(self.visit(elt))
        return tuple(results)

    def visit_keyword(self, node: ast.keyword):
        return {node.arg: self.visit(node.value)}

    def visit_Return(self, node: ast.Return):
        self.visiting_return = True
        results = self.visit(node.value)
        self.visiting_return = False
        self.return_names = results
        if not isinstance(results, tuple):
            results = (results,)
        for res in results:
            try:
                traced_tensor = self.example_inputs[res]
            except KeyError:
                raise RuntimeError(f"Input for {res} is not provided") from e
            self.mark_output(res, traced_tensor)

    def visit_BinOp(self, node: ast.BinOp):
        if self.visiting_return:
            raise SyntaxError("Return value cannot be an expression")
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        op = self.ast_op_bindings(type(node.op))

        # all elements of args should be same for a single compute node.
        input_element = self.get_node(lhs).metadata.element
        name = self.add_compute_node(op, input_element)
        self.add_edge(lhs, name, pos=0)
        self.add_edge(rhs, name, pos=1)
        return name

    def visit_Assign(self, node: ast.BinOp):
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        # Create store node
        input_element = self.get_node(value).metadata.element
        self.add_store_node(input_element, target)
        self.add_edge(value, target)
        return target

    def visit_Call(self, node: ast.Call):
        if self.visiting_return:
            raise SyntaxError("Return value cannot be an expression")
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        if func == "constant":
            # constant ops look like constant(value, DataType.value)
            name = self.add_constant_node(args[0], args[1])
            return name

        if func == "cast":
            # cast ops look like cast(input, dst_type: DataType.value, src_type: DataType.value)
            name = self.add_cast_node(args[1], args[2])
            self.add_edge(args[0], name, pos=0)
            return name

        op = self.ast_op_bindings(func)
        # all elements of args should be same for a single compute node.
        input_element = self.get_node(args[0]).metadata.element
        name = self.add_compute_node(op, input_element)

        # add edges
        for idx, arg in enumerate(args):
            self.add_edge(arg, name, pos=idx)
        return name


class CatlassTensor:
    def __init__(self, element, shape, stride):
        self.element = element
        self.shape = shape
        self.stride = stride


def create_example_tensors(
    var_name_to_buffer_name: dict[str, str],
    name_to_buffer: dict[str, Buffer],
    size_hint_fn: Callable[[Union[Expr, int]], int],
) -> dict[str, CatlassTensor]:
    def catlass_tensor_from_buffer(buffer: Buffer) -> CatlassTensor:
        shape = tuple(buffer.get_layout().size)
        stride = tuple(buffer.get_layout().stride)
        return CatlassTensor(
            element=torch_dtype_to_catlass_type(buffer.get_layout().dtype),
            shape=shape,
            stride=stride,
        )

    return {
        key: catlass_tensor_from_buffer(name_to_buffer[name])
        for key, name in var_name_to_buffer_name.items()
    }


class EVGArgRenames:
    """Handles mapping buffer names to variable names in the cpp kernel signature and body"""

    def __init__(self) -> None:
        self.buf_renames: dict[str, str] = {}

    def new_name(self, name: str) -> str:
        if name in self.buf_renames:
            return self.buf_renames[name]
        else:
            new_name = f"ptr_{len(self.buf_renames)}"
            self.buf_renames[name] = new_name
            return new_name

    def get(self, name: str) -> str:
        return self.buf_renames.get(name)


def trace(
    fn_src: str,
    example_inputs,
    accum_type: DataType,
    output_type: DataType,
    tile_description: TileDescription,
    name_to_buffer: dict[str, Buffer],
    size_hint_fn: Callable[[Union[Expr, int]], int],
    **kwargs: dict[str, Any],
) -> tuple[str, str, str, EVGArgRenames]:
    # Transfer python_func to a DAG
    parser = PythonEVGParser()
    parser.parse(fn_src, example_inputs)

    # Generate Catlass EVG definition
    evg_def = EVGDef(parser)
    evg_str, callback_name = evg_def.definition()

    # Generate Catlass EVG arguments
    evg_arg = EVGArg(parser)
    evg_args, arg_renames = evg_arg.generate_graph_args()

    # Generate compute node length, which is used as "vector tiling"
    evg_compute_length = evg_arg.generate_compute_length()

    # combine evg args and evg_compute_length as the entire args
    evg_args += evg_compute_length

    return callback_name, evg_args, evg_str, arg_renames
