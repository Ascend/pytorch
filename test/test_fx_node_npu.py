# Owner(s): ["module: fx"]
"""
Add validation cases for torch.fx.node APIs on NPU:
1. PyTorch community lacks dedicated test cases for
   torch.fx.node._get_qualified_name, so this file is added.
2. This file validates torch.fx.node._get_qualified_name.
   torch.fx.Node already has comprehensive tests in
   PyTorch upstream test/test_fx.py and requires no NPU adaptation
   (pure graph IR, no device dependency).
"""
import operator

import torch

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx.node import _get_qualified_name


class TestFXNodeGetQualifiedName(TestCase):
    """Test torch.fx.node._get_qualified_name function."""

    def test_builtin_function(self):
        """Test _get_qualified_name with builtin functions."""
        result = _get_qualified_name(getattr)
        self.assertEqual(result, "getattr")

    def test_torch_function(self):
        """Test _get_qualified_name with torch module functions."""
        result = _get_qualified_name(torch.relu)
        self.assertEqual(result, "torch.relu")

    def test_torch_binary_function(self):
        """Test _get_qualified_name with torch namespace function."""
        result = _get_qualified_name(torch.add)
        self.assertEqual(result, "torch.add")

    def test_operator_function(self):
        """Test _get_qualified_name with operator module functions."""
        result = _get_qualified_name(operator.add)
        self.assertEqual(result, "_operator.add")

    def test_tensor_method(self):
        """Test _get_qualified_name with torch.Tensor methods."""
        result = _get_qualified_name(torch.Tensor.add)
        self.assertEqual(result, "torch.Tensor.add")

    def test_submodule_function(self):
        """Test _get_qualified_name with sub-module function."""
        result = _get_qualified_name(torch.nn.functional.relu)
        self.assertEqual(result, "torch.nn.functional.relu")

    def test_consistency_on_repeated_calls(self):
        """Test _get_qualified_name returns consistent results."""
        results = [_get_qualified_name(torch.abs) for _ in range(5)]
        for r in results:
            self.assertEqual(r, "torch.abs")


if __name__ == "__main__":
    run_tests()
