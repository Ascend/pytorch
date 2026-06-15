"""
Add validation cases for torch.fx.node APIs on NPU:
1. PyTorch community lacks direct test cases for the following APIs,
   so this file is added.
2. This file validates torch.fx.node._type_repr, torch.fx.Node.kwargs,
   torch.fx.node.map_arg, torch.fx.Node.next, torch.fx.Node.prev (extendable).
"""

import operator
import types

import torch
import torch.fx
from torch.fx.node import _type_repr, map_arg
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFxNodeExtraApis(TestCase):

    def test_type_repr_builtin_type(self):
        # builtin types return qualname only, no module prefix
        self.assertEqual(_type_repr(int), "int")
        self.assertEqual(_type_repr(float), "float")
        self.assertEqual(_type_repr(str), "str")
        self.assertEqual(_type_repr(bool), "bool")
        self.assertEqual(_type_repr(type(None)), "NoneType")

    def test_type_repr_non_builtin_type(self):
        # non-builtin types return module.qualname
        self.assertEqual(_type_repr(torch.Tensor), "torch.Tensor")
        self.assertEqual(_type_repr(torch.nn.Linear), "torch.nn.modules.linear.Linear")

    def test_type_repr_ellipsis(self):
        # Ellipsis object returns "..."
        self.assertEqual(_type_repr(...), "...")

    def test_type_repr_function(self):
        # FunctionType returns function.__name__
        def my_func():
            pass

        self.assertEqual(_type_repr(my_func), "my_func")

    def test_type_repr_generic_alias(self):
        # GenericAlias (e.g. list[int]) falls through to repr()
        # not treated as a plain type, so module.qualname logic is skipped
        result = _type_repr(list[int])
        self.assertEqual(result, repr(list[int]))

    def test_type_repr_other(self):
        # Non-type objects fall back to repr()
        self.assertEqual(_type_repr(42), "42")
        self.assertEqual(_type_repr("hello"), "'hello'")

    def test_node_kwargs_getter(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.relu, kwargs={"input": x})
        graph.output(node)

        # getter returns the kwargs dict
        self.assertEqual(node.kwargs, {"input": x})
        self.assertIsInstance(node.kwargs, dict)

    def test_node_kwargs_setter_updates_use_def(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        node = graph.call_function(torch.relu, kwargs={"input": x})
        graph.output(node)

        # x should be in node's users before reassignment
        self.assertIn(node, x.users)
        self.assertNotIn(node, y.users)

        # reassign kwargs: x is removed, y is added
        node.kwargs = {"input": y}
        self.assertEqual(node.kwargs, {"input": y})
        self.assertNotIn(node, x.users)
        self.assertIn(node, y.users)

    def test_node_kwargs_setter_empty(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.call_function(torch.relu, kwargs={"input": x})
        graph.output(node)

        node.kwargs = {}
        self.assertEqual(node.kwargs, {})
        # x no longer used by node
        self.assertNotIn(node, x.users)

    def test_map_arg_applies_fn_to_nodes(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        add = graph.call_function(operator.add, args=(x, y))
        graph.output(add)

        visited = []

        def collect(node):
            visited.append(node.name)
            return node

        map_arg(add.args, collect)
        self.assertEqual(visited, ["x", "y"])

    def test_map_arg_non_node_passthrough(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        # non-Node elements pass through unchanged
        result = map_arg((x, 42, "const", 3.14), lambda n: n)
        self.assertIs(result[0], x)
        self.assertEqual(result[1], 42)
        self.assertEqual(result[2], "const")
        self.assertAlmostEqual(result[3], 3.14)

    def test_map_arg_nested_structure(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")

        visited = []

        def collect(node):
            visited.append(node.name)
            return node

        # nested list containing nodes
        map_arg([x, [y]], collect)
        self.assertIn("x", visited)
        self.assertIn("y", visited)

    def test_map_arg_dict(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        visited = []

        def collect(node):
            visited.append(node.name)
            return node

        map_arg({"input": x, "scale": 2.0}, collect)
        self.assertEqual(visited, ["x"])

    def test_map_arg_requires_callable(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        with self.assertRaises(AssertionError):
            map_arg(x, "not_a_callable")

    def test_node_next(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, args=(x,))
        out = graph.output(relu)

        self.assertIs(x.next, relu)
        self.assertIs(relu.next, out)

    def test_node_next_after_append(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, args=(x,))
        neg = graph.call_function(torch.neg, args=(x,))
        graph.output(relu)

        # move neg to after relu
        relu.append(neg)
        self.assertIs(relu.next, neg)

    def test_node_prev(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, args=(x,))
        out = graph.output(relu)

        self.assertIs(relu.prev, x)
        self.assertIs(out.prev, relu)

    def test_node_prev_after_prepend(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, args=(x,))
        neg = graph.call_function(torch.neg, args=(x,))
        graph.output(relu)

        # move neg to before relu
        relu.prepend(neg)
        self.assertIs(relu.prev, neg)

    def test_node_next_prev_consistent(self):
        # next and prev are inverses of each other
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, args=(x,))
        graph.output(relu)

        self.assertIs(x.next.prev, x)
        self.assertIs(relu.prev.next, relu)


if __name__ == "__main__":
    run_tests()
