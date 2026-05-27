# Owner(s): ["module: fx"]
"""
Test PyTorch FX symbolic_shapes APIs on NPU (PyTorch 2.7-2.11).

Upstream test/dynamo/test_misc.py::test_shape_env_recorded_function_fallback
only calls constrain_range/constrain_unify with Python int for ShapeEnv fallback;
it cannot validate NPU device context or NPU tensor paths.

Upstream test/ has no coverage for compute_unbacked_bindings, ConvertIntKey, or
ConvertIntKey.get on NPU tensors. This file adds the missing NPU-side tests.

This test verifies:

* torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings
* torch.fx.experimental.symbolic_shapes.constrain_range
* torch.fx.experimental.symbolic_shapes.constrain_unify
* torch.fx.experimental.symbolic_shapes.ConvertIntKey
* torch.fx.experimental.symbolic_shapes.ConvertIntKey.get
"""

import torch
import torch_npu
import torch.fx.experimental.symbolic_shapes as symbolic_shapes
from torch.testing._internal.common_utils import TestCase, run_tests


class TestFXExperimentalSymbolic(TestCase):
    """Verify symbolic_shapes helpers on NPU."""

    def test_compute_unbacked_bindings(self):
        x = torch.randn(2, 3, device="npu")
        result = symbolic_shapes.compute_unbacked_bindings(None, x)
        self.assertTrue(result is None or isinstance(result, dict))

    def test_compute_unbacked_bindings_npu_context(self):
        with torch.npu.device(0):
            x = torch.randn(4, 5, device="npu")
            result = symbolic_shapes.compute_unbacked_bindings(None, x)
            self.assertTrue(result is None or isinstance(result, dict))

    def test_compute_unbacked_bindings_to_npu(self):
        x = torch.randn(2, 3).to("npu")
        result = symbolic_shapes.compute_unbacked_bindings(None, x)
        self.assertTrue(result is None or isinstance(result, dict))

    def test_constrain_range(self):
        symbolic_shapes.constrain_range(5, min=2, max=10)
        with self.assertRaises(ValueError):
            symbolic_shapes.constrain_range(1, min=2, max=10)

    def test_constrain_range_npu_context(self):
        with torch.npu.device(0):
            symbolic_shapes.constrain_range(5, min=2, max=10)

    def test_constrain_unify(self):
        symbolic_shapes.constrain_unify(5, 5)
        with self.assertRaises(AssertionError):
            symbolic_shapes.constrain_unify(5, 6)

    def test_constrain_unify_npu_context(self):
        with torch.npu.device(0):
            symbolic_shapes.constrain_unify(7, 7)

    def test_convert_int_key_singleton(self):
        cik = symbolic_shapes.ConvertIntKey()
        self.assertEqual(cik.get(True), 1)
        self.assertEqual(cik.get(False), 0)

    def test_convert_int_key_get_npu_derived(self):
        cik = symbolic_shapes.ConvertIntKey()
        x = torch.tensor([1, 2, 3], device="npu")
        self.assertEqual(cik.get(bool((x[0] == x[0]).item())), 1)
        self.assertEqual(cik.get(bool((x[0] != x[1]).item())), 1)
        self.assertEqual(cik.get(bool((x[1] <= x[0]).item())), 0)

    def test_convert_int_key_npu_context(self):
        with torch.npu.device(0):
            cik = symbolic_shapes.ConvertIntKey()
            self.assertEqual(cik.get(True), 1)
            self.assertEqual(cik.get(False), 0)


if __name__ == "__main__":
    run_tests()
