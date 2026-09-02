"""
Add validation cases for torch.compiler.compile (extendable):
This file validates torch.compiler.compile with basic functional tests.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class TestTorchCompilerCompile(TestCase):

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def test_torch_compiler_compile_exists(self):
        """Validate torch.compiler.compile is available and callable."""
        self.assertTrue(hasattr(torch.compiler, "compile"))
        self.assertTrue(callable(torch.compiler.compile))

    def test_torch_compiler_compile_basic(self):
        """Validate torch.compiler.compile works with a simple function."""
        def fn(x):
            return x + 1

        x = torch.randn(3, 3)
        opt_fn = torch.compiler.compile(fn, backend="eager")
        result = opt_fn(x)
        self.assertTrue(torch.allclose(result, fn(x)))

    def test_torch_compiler_compile_with_model(self):
        """Validate torch.compiler.compile works with nn.Module."""
        model = ToyModel()
        x = torch.randn(1, 10)
        opt_model = torch.compiler.compile(model, backend="eager")
        result = opt_model(x)
        expected = model(x)
        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    run_tests()
