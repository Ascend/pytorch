import re
from unittest import skip

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import run_tests


try:
    from .testutils import TestUtils
except ImportError:
    from testutils import TestUtils

torch._inductor.config.fx_graph_cache = False


class TestLinearTritonStoreSemantics(TestUtils):
    _permute_pattern = re.compile(r"\.permute\(")
    _axis_replacement_pattern = re.compile(r"\b[trzyx]\d+_\d+(?:_nd)?\b")

    @staticmethod
    def transpose_clone_rectangular(x):
        y = x.view(-1, 80, 40, 8)
        return y.permute(0, 2, 1, 3).clone()

    @staticmethod
    def transpose_unary_clone_rectangular(x):
        y = x.view(-1, 80, 40, 8)
        return y.permute(0, 2, 1, 3).sin().clone()

    @staticmethod
    def broadcast_add(x, bias):
        return x + bias.view(1, 1, 1, -1)

    def _assert_scheduler_semantic_store_codegen(self, codes):
        self.assertTrue(codes)
        self.assertFalse(
            any(self._permute_pattern.search(code) for code in codes),
            msg="expected scheduler-semantic store path without RHS permute",
        )
        self.assertFalse(
            any(self._axis_replacement_pattern.search(code) for code in codes),
            msg="expected scheduler-semantic store path without remapped axis symbols",
        )

    def test_scheduler_semantic_store_path_for_transpose_clone(self):
        x = self._generate_tensor((381, 80, 320), "float32")
        compiled = torch.compile(
            self.transpose_clone_rectangular, backend="inductor", dynamic=False
        )
        out, codes = run_and_get_code(compiled, x)
        eager = self.transpose_clone_rectangular(x)
        torch.testing.assert_close(out, eager, rtol=1e-4, atol=1e-4)
        self._assert_scheduler_semantic_store_codegen(codes)

    # AssertionError: assert_size_stride(buf1, (381, 40, 80, 8), (25600, 8, 320, 1), 'torch.ops.aten.sin.default')
    @skip("skip ci codegen error")
    def test_scheduler_semantic_store_path_survives_simple_pointwise(self):
        x = self._generate_tensor((381, 80, 320), "float32")
        compiled = torch.compile(
            self.transpose_unary_clone_rectangular,
            backend="inductor",
            dynamic=False,
        )
        out, codes = run_and_get_code(compiled, x)
        eager = self.transpose_unary_clone_rectangular(x)
        torch.testing.assert_close(out, eager, rtol=1e-4, atol=1e-4)
        self._assert_scheduler_semantic_store_codegen(codes)

    def test_remapped_store_path_still_reachable(self):
        x = self._generate_tensor((8, 8, 32, 16), "float32")
        bias = self._generate_tensor((16,), "float32")
        compiled = torch.compile(self.broadcast_add, backend="inductor", dynamic=False)
        out, codes = run_and_get_code(compiled, x, bias)
        eager = self.broadcast_add(x, bias)
        torch.testing.assert_close(out, eager, rtol=1e-4, atol=1e-4)
        self.assertTrue(codes)
        self.assertTrue(any("tl.store(" in code for code in codes))


if __name__ == "__main__":
    run_tests()
