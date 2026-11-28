import re
import unittest

import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code

import torch_npu  # noqa: F401


@unittest.skipUnless(torch.npu.is_available(), "requires NPU")
class TestPersistentSumPadding(unittest.TestCase):
    def _check_embedding_variance(self, width, block):
        from torch_npu._inductor import config as npu_config

        if not npu_config.is_ascend950:
            self.skipTest("requires the Ascend 950 unified SIMT path")

        def normalize(indices, table, weight, bias):
            indices = torch.ops.aten.reshape.default(indices, [-1])
            x = torch.ops.aten.embedding.default(table, indices).unsqueeze(1)
            variance = torch.ops.aten.var.correction(x, [-1], correction=0, keepdim=True)
            scale = torch.rsqrt(variance + 1e-5) * weight
            scaled_x = x * scale
            mean = torch.ops.aten.mean.dim(x, [-1], True)
            return scaled_x + (bias - mean * scale)

        generator = torch.Generator().manual_seed(0)
        indices = torch.randint(8193, (60, 64), generator=generator)
        # A nonzero mean makes squared deviations in padded lanes nonzero.
        table = torch.empty(8193, width).normal_(1.0, 0.2, generator=generator)
        args = indices, table, torch.ones(1, 1, width), torch.zeros(1, 1, width)
        with torch.no_grad(), config.patch(
            force_disable_caches=True, max_autotune=True,
            compile_threads=1, cpp_wrapper=False,
        ):
            expected = normalize(*args)
            inputs = tuple(tensor.npu() for tensor in args)
            compiled = torch.compile(normalize, fullgraph=True, dynamic=False)
            actual, sources = run_and_get_code(compiled, *inputs)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)

        source = "\n".join(sources)
        self.assertIn("@triton_heuristics.persistent_reduction(", source)
        self.assertRegex(source, rf"R\d+BLOCK_SUB: tl.constexpr = {block}\b")
        if width < block:
            # Both the mean and squared-deviation sums must exclude padding.
            operands = re.findall(r"\btl\.sum\((\w+), 1\)", source)
            self.assertEqual(len(operands), 2)
            for operand in operands:
                self.assertRegex(
                    source,
                    rf"\b{operand} = tl\.where\([^\n]*\br\d+_mask\b[^\n]*, 0\)",
                )
        # A full R block may omit a mask proven to be constant True.

    def test_embedding_variance_padded_reduction(self):
        self._check_embedding_variance(1536, 2048)

    def test_embedding_variance_full_reduction_block(self):
        self._check_embedding_variance(1024, 1024)


if __name__ == "__main__":
    unittest.main()
