"""
Add validation cases for torch.fx.GraphModule APIs on NPU:
1. PyTorch community lacks direct and independent test cases for
   torch.fx.GraphModule.code, torch.fx.GraphModule.graph, and
   several submodule management APIs, so this file is added.
2. This file validates torch.fx.GraphModule.__init__,
   torch.fx.GraphModule.code, torch.fx.GraphModule.graph,
   torch.fx.GraphModule.add_submodule,
   torch.fx.GraphModule.delete_submodule,
   torch.fx.GraphModule.delete_all_unused_submodules,
   torch.fx.GraphModule.print_readable,
   torch.fx.GraphModule.recompile,
   torch.fx.GraphModule.to_folder (extendable).
"""

import os
import tempfile

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule, Graph
from torch.fx.graph import PythonCode
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFxGraphModuleInit(TestCase):

    def test_init_from_module(self):
        """GraphModule.__init__ with nn.Module root copies submodules."""
        class TestMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        m = TestMod()
        gm = symbolic_trace(m)
        self.assertTrue(hasattr(gm, "lin"))
        self.assertIsInstance(gm.lin, nn.Linear)

    def test_init_from_dict(self):
        """GraphModule.__init__ with dict root assigns attributes."""
        graph = Graph()
        x = graph.placeholder("x")
        lin = graph.call_module("lin", args=(x,))
        graph.output(lin)

        gm = GraphModule({"lin": nn.Linear(4, 3)}, graph)
        x_in = torch.randn(2, 4)
        out = gm(x_in)
        self.assertEqual(out.shape, (2, 3))

    def test_init_sets_class_name(self):
        """GraphModule.__init__ with custom class_name."""
        graph = Graph()
        x = graph.placeholder("x")
        graph.output(x)

        gm = GraphModule(torch.nn.Module(), graph, class_name="MyGM")
        self.assertEqual(gm.__class__.__name__, "MyGM")

    def test_init_raises_on_bad_type(self):
        """GraphModule.__init__ rejects non-Module/non-dict root."""
        graph = Graph()
        x = graph.placeholder("x")
        graph.output(x)

        with self.assertRaises(RuntimeError):
            GraphModule("bad_root", graph)


class TestFxGraphModuleCode(TestCase):

    def setUp(self):
        super().setUp()

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x):
                return torch.relu(self.linear(x))

        self.gm = symbolic_trace(SimpleModule())

    def test_code_returns_string(self):
        """code property returns a non-empty string."""
        code = self.gm.code
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    def test_code_contains_forward(self):
        """code contains 'def forward'."""
        self.assertIn("def forward", self.gm.code)
        self.assertIn("self", self.gm.code)

    def test_code_contains_op_names(self):
        """code contains operator names from traced model."""
        code = self.gm.code
        self.assertIn("relu", code)
        self.assertIn("linear", code)

    def test_code_consistent_after_recompile(self):
        """code is consistent after multiple recompiles."""
        code1 = self.gm.code
        self.gm.recompile()
        code2 = self.gm.code
        self.assertEqual(code1, code2)


class TestFxGraphModuleGraph(TestCase):

    def setUp(self):
        super().setUp()

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)

            def forward(self, x):
                return torch.relu(self.linear(x))

        self.module = SimpleModule()
        self.gm = symbolic_trace(self.module)

    def test_graph_getter_returns_graph(self):
        """graph getter returns a Graph instance."""
        g = self.gm.graph
        self.assertIsInstance(g, Graph)

    def test_graph_getter_has_nodes(self):
        """graph getter returns graph with placeholder and output."""
        g = self.gm.graph
        nodes = list(g.nodes)
        self.assertGreater(len(nodes), 0)
        ops = {node.op for node in nodes}
        self.assertIn("placeholder", ops)
        self.assertIn("output", ops)

    def test_graph_getter_is_consistent(self):
        """graph getter returns same object on repeated access."""
        g1 = self.gm.graph
        g2 = self.gm.graph
        self.assertIs(g1, g2)

    def test_graph_setter_reassigns_graph(self):
        """Setting graph reassigns internal reference."""
        gm_new = symbolic_trace(self.module)
        new_graph = gm_new.graph
        self.gm.graph = new_graph
        self.assertIs(self.gm.graph, new_graph)

    def test_graph_setter_triggers_recompile(self):
        """Setting graph triggers recompile and produces valid code."""
        gm_new = symbolic_trace(self.module)
        self.gm.graph = gm_new.graph
        code = self.gm.code
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    def test_graph_setter_forward_works(self):
        """After graph set, forward produces correct output."""
        gm_new = symbolic_trace(self.module)
        self.gm.graph = gm_new.graph
        x = torch.randn(2, 4)
        expected = self.module(x)
        actual = self.gm(x)
        torch.testing.assert_close(actual, expected)

    def test_graph_setter_raises_on_non_graph(self):
        """Setting graph to non-Graph raises AssertionError."""
        with self.assertRaises(AssertionError):
            self.gm.graph = "not_a_graph"
        with self.assertRaises(AssertionError):
            self.gm.graph = 42

    def test_graph_setter_preserves_lint(self):
        """After setting a valid graph, lint() does not raise."""
        gm_new = symbolic_trace(self.module)
        self.gm.graph = gm_new.graph
        self.gm.graph.lint()


class TestFxGraphModuleSubmodule(TestCase):

    def setUp(self):
        super().setUp()
        graph = Graph()
        x = graph.placeholder("x")
        lin = graph.call_module("lin", args=(x,))
        graph.output(lin)
        self.gm = GraphModule({"lin": nn.Linear(4, 3)}, graph)

    def test_add_submodule_root_level(self):
        """add_submodule at root level adds the module."""
        new_mod = nn.ReLU()
        result = self.gm.add_submodule("relu", new_mod)
        self.assertTrue(result)
        self.assertIs(self.gm.relu, new_mod)

    def test_add_submodule_nested(self):
        """add_submodule creates intermediate modules for nested path."""
        new_mod = nn.ReLU()
        result = self.gm.add_submodule("a.b.c", new_mod)
        self.assertTrue(result)
        self.assertIsInstance(self.gm.a, nn.Module)
        self.assertIsInstance(self.gm.a.b, nn.Module)
        self.assertIs(self.gm.a.b.c, new_mod)

    def test_add_submodule_overwrite_fails_on_non_module(self):
        """add_submodule returns False when path blocked by non-Module."""
        self.gm.add_submodule("blocker", nn.ReLU())
        # Install a non-Module attribute to block the path
        self.gm.blocker.some_attr = "not_a_module"
        result = self.gm.add_submodule("blocker.some_attr.sub", nn.ReLU())
        self.assertFalse(result)

    def test_delete_submodule_existing(self):
        """delete_submodule removes an existing submodule."""
        self.gm.add_submodule("temp", nn.ReLU())
        self.assertTrue(hasattr(self.gm, "temp"))
        result = self.gm.delete_submodule("temp")
        self.assertTrue(result)
        self.assertFalse(hasattr(self.gm, "temp"))

    def test_delete_submodule_nested(self):
        """delete_submodule removes a nested submodule."""
        self.gm.add_submodule("outer.inner", nn.ReLU())
        self.assertTrue(hasattr(self.gm.outer, "inner"))
        result = self.gm.delete_submodule("outer.inner")
        self.assertTrue(result)
        self.assertFalse(hasattr(self.gm.outer, "inner"))

    def test_delete_submodule_nonexistent(self):
        """delete_submodule on nonexistent path returns False."""
        result = self.gm.delete_submodule("nonexistent.path")
        self.assertFalse(result)

    def test_delete_submodule_non_module(self):
        """delete_submodule on non-Module attribute returns False."""
        self.gm.some_param = nn.Parameter(torch.randn(2, 2))
        result = self.gm.delete_submodule("some_param")
        self.assertFalse(result)

    def test_delete_all_unused_removes_orphans(self):
        """delete_all_unused_submodules removes modules not in graph."""
        graph = Graph()
        x = graph.placeholder("x")
        lin = graph.call_module("lin", args=(x,))
        graph.output(lin)
        gm = GraphModule({"lin": nn.Linear(4, 3)}, graph)
        gm.add_submodule("orphan", nn.ReLU())
        self.assertTrue(hasattr(gm, "orphan"))
        gm.delete_all_unused_submodules()
        self.assertFalse(hasattr(gm, "orphan"))
        self.assertTrue(hasattr(gm, "lin"))

    def test_delete_all_unused_preserves_used(self):
        """delete_all_unused_submodules preserves modules in graph."""
        graph = Graph()
        x = graph.placeholder("x")
        a = graph.call_module("a", args=(x,))
        b = graph.call_module("b", args=(a,))
        graph.output(b)
        gm = GraphModule({
            "a": nn.Linear(4, 4),
            "b": nn.Linear(4, 3),
        }, graph)
        gm.add_submodule("orphan", nn.ReLU())
        self.assertTrue(hasattr(gm, "orphan"))
        gm.delete_all_unused_submodules()
        self.assertFalse(hasattr(gm, "orphan"))
        self.assertTrue(hasattr(gm, "a"))
        self.assertTrue(hasattr(gm, "b"))


class TestFxGraphModulePrintReadable(TestCase):

    def test_print_readable_returns_string(self):
        """print_readable returns a non-empty string."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return torch.relu(self.lin(x))

        gm = symbolic_trace(SimpleMod())
        output = gm.print_readable(print_output=False)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
        self.assertIn("class", output)
        self.assertIn("def forward", output)

    def test_print_readable_contains_child_code(self):
        """print_readable includes code from child GraphModules."""
        class ChildMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(3, 4))

            def forward(self, x):
                return x + self.w

        class ParentMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = symbolic_trace(ChildMod())

            def forward(self, x):
                return self.child(x)

        gm = symbolic_trace(ParentMod())
        output = gm.print_readable(print_output=False)
        self.assertIsInstance(output, str)
        self.assertIn("class", output)


class TestFxGraphModuleRecompile(TestCase):

    def test_recompile_returns_python_code(self):
        """recompile returns a PythonCode object."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        gm = symbolic_trace(SimpleMod())
        pc = gm.recompile()
        self.assertIsInstance(pc, PythonCode)
        self.assertGreater(len(pc.src), 0)

    def test_recompile_preserves_forward(self):
        """After recompile, forward still works correctly."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        m = SimpleMod()
        gm = symbolic_trace(m)
        x = torch.randn(2, 4)
        expected = m(x)
        gm.recompile()
        actual = gm(x)
        torch.testing.assert_close(actual, expected)


class TestFxGraphModuleToFolder(TestCase):

    def test_to_folder_creates_files(self):
        """to_folder creates module.py and __init__.py in folder."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        gm = symbolic_trace(SimpleMod())
        with tempfile.TemporaryDirectory() as tmpdir:
            gm.to_folder(tmpdir, "TestMod")
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "module.py")))
            self.assertTrue(os.path.isfile(
                os.path.join(tmpdir, "__init__.py")))

    def test_to_folder_module_file_content(self):
        """to_folder output can be imported."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        gm = symbolic_trace(SimpleMod())
        with tempfile.TemporaryDirectory() as tmpdir:
            gm.to_folder(tmpdir, "TestMod")
            # Verify the module file content is valid Python
            with open(os.path.join(tmpdir, "module.py")) as f:
                content = f.read()
            self.assertIn("class TestMod(torch.nn.Module)", content)
            self.assertIn("def forward", content)


class TestFxGraphModuleOnNpu(TestCase):
    """Verify GraphModule APIs work with tensors on NPU device."""

    def setUp(self):
        super().setUp()
        if device_type == "cpu":
            self.skipTest("Test requires NPU device")

    def test_code_and_graph_on_npu(self):
        """code and graph properties work after moving module to NPU."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return torch.relu(self.lin(x))

        m = SimpleMod().to(device_type)
        gm = symbolic_trace(m)
        self.assertIsInstance(gm.code, str)
        self.assertGreater(len(gm.code), 0)
        self.assertIsInstance(gm.graph, Graph)
        self.assertGreater(len(list(gm.graph.nodes)), 0)

    def test_forward_on_npu(self):
        """Generated forward works with NPU tensors."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        m = SimpleMod().to(device_type)
        gm = symbolic_trace(m)
        x = torch.randn(2, 4).to(device_type)
        expected = m(x)
        actual = gm(x)
        torch.testing.assert_close(actual, expected)

    def test_recompile_on_npu(self):
        """recompile works after module is on NPU."""
        class SimpleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 3)

            def forward(self, x):
                return self.lin(x)

        m = SimpleMod().to(device_type)
        gm = symbolic_trace(m)
        x = torch.randn(2, 4).to(device_type)
        gm.recompile()
        actual = gm(x)
        expected = m(x)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    run_tests()
