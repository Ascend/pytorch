"""filter_masks must keep the mask guarding a load over a dynamically sized cat.

ops.masked ands one <axis>_mask into the body per loop axis of the masked
subblock and skips size symbols, since "y1 < s0" constrains y1 and not the
symbol itself. A size symbol left in current_subblock_axis therefore contributed
no mask while still failing the subset test in filter_masks for every load whose
index did not spell it out, and the tmp mask guarding that load was dropped. The
first slice of a cat over a dynamic dimension then read out of bounds.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

import torch_npu
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.codegen import triton as triton_codegen

if not torch.npu.is_available():
    raise unittest.SkipTest("NPU is not available")

device = "npu"


class TestDynamicCatMask(TestCase):
    """filter_masks may drop a redundant axis mask, never a semantic one."""

    @staticmethod
    def _axis(name):
        return SimpleNamespace(
            name=name,
            is_vectorized_split=False,
            is_tiling_axis=True,
            is_reduction=False,
            is_no_loop_axis=False,
        )

    def _filter_masks(self, subblock_axis, index_vars):
        """Run filter_masks over a two axis kernel and report what survived.

        The mask set holds the two axis masks plus one tmp mask standing in for
        the guard a masked subblock puts on its load.
        """
        kernel = object.__new__(triton_codegen.NPUIndexTritonKernel)
        kernel.sorted_axis = [self._axis("x0"), self._axis("y1")]
        kernel.persistent_reduction = False
        kernel.npu_kernel_type = triton_codegen.NPUKernelType.SIMD

        guard = object.__new__(triton_codegen.TritonCSEVariable)
        guard.name = "tmp3"
        mask_vars = {guard, "x0_mask", "y1_mask"}  # noqa: set_linter

        virtualized = SimpleNamespace(
            kernel=SimpleNamespace(current_subblock_axis=set(subblock_axis))
        )
        with (
            patch.object(triton_codegen, "V", virtualized),
            patch.object(triton_codegen, "get_allow_dynamic", return_value=True),
        ):
            kernel.filter_masks(mask_vars, index_vars)
        return {str(mask_var) for mask_var in mask_vars}

    @parametrize("size_symbol", ("s0", "ps0", "i0"))
    def test_size_symbol_alone_keeps_the_guarding_mask(self, size_symbol):
        # What a cat over a dynamic dimension records: the masked subblock is
        # indexed by the size symbol, while the load reads the first slice and
        # never mentions it.
        kept = self._filter_masks({size_symbol}, ["y1"])

        self.assertIn("tmp3", kept)

    def test_size_symbol_beside_a_loop_axis_keeps_the_guarding_mask(self):
        kept = self._filter_masks({"y1", "s0"}, ["y1"])

        self.assertIn("tmp3", kept)
        # y1 is a subblock axis, so ops.masked already anded y1_mask into the
        # body and the axis mask on the load is the redundant kind.
        self.assertEqual(kept, {"tmp3", "x0_mask"})

    def test_loop_axis_outside_the_index_still_drops_the_guarding_mask(self):
        # x0 is a real loop axis that this load does not index, so the guard was
        # built for a different shape and must not travel with it.
        kept = self._filter_masks({"x0"}, ["y1"])

        self.assertEqual(kept, {"x0_mask", "y1_mask"})

    def test_no_subblock_leaves_every_mask_alone(self):
        kept = self._filter_masks(set(), ["y1"])

        self.assertEqual(kept, {"tmp3", "x0_mask", "y1_mask"})

    def test_dynamic_cat_slice_stays_in_bounds(self):
        def fn(head, tail):
            return torch.cat([head, tail], dim=1) + 1.0

        head = torch.randn(32, 200, device=device)
        tail = torch.randn(32, 56, device=device)
        torch._dynamo.mark_dynamic(head, 1)
        expected = fn(head, tail)

        try:
            compiled = torch.compile(fn, backend="inductor", dynamic=True)
            torch.testing.assert_close(compiled(head, tail), expected)

            # A second extent reuses the compiled kernel, so the guard has to
            # follow the symbol rather than the size that was traced.
            wider = torch.randn(32, 328, device=device)
            torch.testing.assert_close(compiled(wider, tail), fn(wider, tail))
        finally:
            torch._dynamo.reset()


instantiate_parametrized_tests(TestDynamicCatMask)


if __name__ == "__main__":
    run_tests()
