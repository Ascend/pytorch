# Owner(s): ["module: fx"]

import torch
from torch.fx import symbolic_trace
from torch.fx.graph import CodeGen, PythonCode
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFXCodegenAPI(TestCase):
    def test_graph_python_code_returns_python_code(self):
        def fn(x, y):
            return torch.relu(x + y)

        gm = symbolic_trace(fn)
        python_code = gm.graph.python_code("self")

        self.assertIsInstance(python_code, PythonCode)
        self.assertTrue(hasattr(python_code, "src"))
        self.assertTrue(hasattr(python_code, "globals"))
        self.assertIsInstance(python_code.src, str)
        self.assertIsInstance(python_code.globals, dict)
        self.assertIn("def forward", python_code.src)

    def test_graph_set_codegen(self):
        class ListCodeGen(CodeGen):
            def gen_fn_def(self, free_vars, maybe_return_annotation):
                return f"""def forward(self, args_list){maybe_return_annotation}:
    {", ".join(free_vars)} = args_list"""

            def process_inputs(self, *inputs):
                if len(inputs) != 1:
                    raise RuntimeError("Expected exactly one input")
                return inputs[0]

        def fn(x, y):
            return x + y

        gm = symbolic_trace(fn)
        gm.graph.set_codegen(ListCodeGen())
        gm.recompile()

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.assertEqual(gm([x, y]), x + y)


if __name__ == "__main__":
    run_tests()
