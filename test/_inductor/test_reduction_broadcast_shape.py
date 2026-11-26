import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "default")

import torch
from torch._inductor import config
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_npu  # noqa: F401
from version_mark import runIfVersion


@unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
class TestReductionBroadcastShape(TestCase):
    def setUp(self):
        super().setUp()
        # A kernel cached by an earlier run would hide the regression, and the
        # persistent layout is the one that produced tl.full([1], 5, tl.int64).
        self.enterContext(config.patch({
            "force_disable_caches": True,
            "triton.persistent_reductions": True,
        }))

    @runIfVersion(max="2.13")
    def test_div_by_int64_ones_count(self):
        def fn(value):
            # ones_like keeps the count in the graph; torch.ones(value.shape)
            # would be constant-folded away by dynamo and never reach codegen.
            count = torch.ones_like(value, dtype=torch.int64)
            return value.sum() / count.sum()

        value = torch.full((3, 5), 2.0, device="npu", dtype=torch.float32)
        expected = fn(value)
        self.assertEqual(expected.item(), 2.0)

        compiled = torch.compile(fn, backend="inductor", dynamic=False, fullgraph=True)
        try:
            actual = compiled(value)
            torch.npu.synchronize()
            # Before the fix sum_1 was 5 instead of 15, so this was 30 / 5 == 6.0.
            self.assertEqual(actual, expected, atol=1e-6, rtol=0)
        finally:
            torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
