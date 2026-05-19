# Owner(s): ["module: test"]

"""
Add test cases for torch.fx APIs on NPU:

PyTorch community lacks sufficient and direct API validations for some FX internal APIs, so this file is added.
This file validates:
- torch.fx.graph._custom_builtins
- torch.fx.graph._CustomBuiltin
- torch.fx.experimental.symbolic_shapes.SymbolicContext

These APIs are internal to PyTorch FX and have no direct test coverage in community test_fx.py.
All tests are NPU-agnostic (framework-level only).
"""

import torch

from torch.fx.experimental.symbolic_shapes import SymbolicContext
from torch.fx.graph import _custom_builtins, _CustomBuiltin
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFxCustomBuiltins(TestCase):

    def test_custom_builtins_items(self):
        items = _custom_builtins.items()
        self.assertIsNotNone(items)
        items_list = list(items)
        self.assertGreater(len(items_list), 0)
        for k, v in items_list:
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, _CustomBuiltin)

    def test_CustomBuiltin_type(self):
        self.assertIsNotNone(_CustomBuiltin)
        self.assertEqual(
            _CustomBuiltin.__module__,
            "torch.fx.graph",
        )

    def test_CustomBuiltin_instance(self):
        builtin = _custom_builtins["inf"]
        self.assertIsNotNone(builtin)
        self.assertIsInstance(builtin, _CustomBuiltin)

    def test_SymbolicContext_import(self):
        self.assertIsNotNone(SymbolicContext)
        self.assertEqual(
            SymbolicContext.__module__,
            "torch.fx.experimental.symbolic_shapes",
        )

    def test_SymbolicContext_instance(self):
        ctx = SymbolicContext()
        self.assertIsNotNone(ctx)
        self.assertIsInstance(ctx, SymbolicContext)


if __name__ == "__main__":
    run_tests()
    