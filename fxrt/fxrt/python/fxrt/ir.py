"""
A Python wrapper for the fxrt C++ GraphExecutor to build and run computation graphs.
"""
from typing import List
from fxrt._fxrt_ir import (
    GraphExecutor as _GraphExecutor,
    Node,
    Op,
    Tensor,
    Value,
    Tuple,
    SymbolicVar,
    SymbolicConst,
    SymbolicExpr,
    DataType,
    Device,
    DeviceType,
)


class GraphExecutor:
    """
    A Python wrapper for the C++ GraphExecutor to build and run computation graphs.

    This class can be used as a context manager:

    with GraphExecutor("my_graph") as executor:
        ...
        executor.run()
    """

    def __init__(self, name: str = "default_graph"):
        self._executor = _GraphExecutor()
        self._name = name

    def __del__(self):
        # Best-effort cleanup for input static cache keyed by this executor.
        try:
            from fxrt import _fxrt_torch  # pylint: disable=import-outside-toplevel

            _fxrt_torch.clear_graph_input_static_cache(id(self))
        except Exception:  # pylint: disable=broad-except
            pass

    def __enter__(self):
        self._executor.begin_graph(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.end_graph()

    def add_parameter_node(self, value: Value) -> Node:
        """Add a parameter node to the graph."""
        return self._executor.add_parameter_node(value)

    def add_input_node(self, value: Value) -> Node:
        """Add an input node to the graph."""
        return self._executor.add_input_node(value)

    def add_op_node(self, op: Op, inputs: List[Node], output: Value) -> Node:
        """Add an operation node to the graph."""
        if op == Op.tuple_getitem:
            output = self._tuple_getitem(inputs)
        return self._executor.add_op_node(op, inputs, output)

    def _tuple_getitem(self, inputs: List[Node]) -> Value:
        """Get a value from a tuple, only used for Op.tuple_getitem."""
        if len(inputs) != 2:
            raise ValueError("Tuple getitem requires exactly 2 inputs")
        return inputs[0].output.to_tuple()[inputs[1].output.to_int()]

    def make_tuple(self, inputs: List[Node]) -> Node:
        """Add a make_tuple operation to the graph."""
        output = Value(Tuple([input.output for input in inputs]))
        node = self._executor.add_op_node(Op.make_tuple, inputs, output)
        return node

    def add_value_node(self, value: Value) -> Node:
        """Add a constant node to the graph from a python object (e.g. torch.Tensor, scalar)."""
        return self._executor.add_value_node(value)

    def add_return_node(self, node: Node):
        """Add a return node to the graph."""
        self._executor.add_return_node(node)

    def run(self, is_dynamic: bool = True) -> Value:
        """Run the built graph and return the output."""
        self._executor.run_graph(is_dynamic)
        return self._executor.get_output()

    def build(self):
        """Optimize the graph and build kernels."""
        # self._executor.opt_graph()
        self._executor.build_executor()

    def dump_graph(self, print_stdout=True):
        """Dump the graph definition."""
        return self._executor.dump_graph(print_stdout)

    def export_ir_graph(self):
        """Export InferRT C++ IR snapshot for FxConverter."""
        return self._executor.export_ir_graph()


# Re-export for convenience
__all__ = [
    "GraphExecutor",
    "Node",
    "Op",
    "Tensor",
    "Value",
    "Tuple",
    "SymbolicVar",
    "SymbolicConst",
    "SymbolicExpr",
    "DataType",
    "Device",
    "DeviceType",
]
