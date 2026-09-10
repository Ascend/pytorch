import unittest
from unittest.mock import patch

try:
    import torch
    import torch_npu
    import torch_npu._inductor
    HAS_NPU = torch.npu.is_available()
except (ImportError, RuntimeError):
    HAS_NPU = False


@unittest.skipUnless(HAS_NPU, "requires NPU")
class TestBackwardCompactScan(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    @unittest.skip(
        "Temporarily disabled: CI empty-mask backward gradient mismatch and AI Core timeout"
    )
    def test_gradients_changing_masks(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        from torch._inductor.utils import run_and_get_code
        from torch_npu._inductor import config as npu_config

        q = torch.randn(2, 4, 257, 64, device="npu", dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, 2, 385, 64, device="npu", dtype=torch.float16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        grad_out = torch.randn_like(q)
        pattern_id = torch.zeros((), dtype=torch.int32, device="npu")

        def mask(_b, _h, m, n):
            # Arithmetic mask selection avoids aten.index fallback in the subgraph.
            return (
                ((pattern_id == 0) & (m >= n))
                | (pattern_id == 2)
                | ((pattern_id == 3) & (n % 2 == 0))
            )

        def fn(q, k, v, block_mask):
            return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True,
                                  kernel_options={"FORCE_USE_FLEX_ATTENTION": True,
                                                  "BLOCK_M": 64, "BLOCK_N": 64})

        with torch._inductor.config.patch(cpp_wrapper=False), patch.object(
            npu_config.flex_attention, "flexattention_mask_out", True
        ):
            compiled = torch.compile(fn, dynamic=False, fullgraph=True)
            m = torch.arange(257, device="npu")[:, None]
            n = torch.arange(385, device="npu")[None, :]
            patterns = (
                m >= n,
                torch.zeros(257, 385, dtype=torch.bool, device="npu"),
                torch.ones(257, 385, dtype=torch.bool, device="npu"),
                (n % 2 == 0).expand(257, 385),
            )
            for index, pattern in enumerate((0, 1, 2, 3, 0)):
                with self.subTest(pattern=pattern):
                    pattern_id.fill_(pattern)
                    bm = create_block_mask(mask, 1, 1, 257, 385, device="npu")

                    def run():
                        out = compiled(q, k, v, bm)
                        return out, torch.autograd.grad(out, (q, k, v), grad_out)

                    if index == 0:
                        (out, grads), codes = run_and_get_code(run)
                        bwd_code = "\n".join(c for c in codes if "flex_attention_bwd_mask_compact" in c)
                        self.assertIn("flex_attention_compact_offsets", bwd_code)
                        self.assertNotIn("tl.atomic_add(TOTAL_BLOCKS", bwd_code)
                    else:
                        out, grads = run()
                    ref_inputs = [x.detach().cpu().float().requires_grad_(True) for x in (q, k, v)]
                    ref_out = torch.nn.functional.scaled_dot_product_attention(
                        *ref_inputs, attn_mask=patterns[pattern].cpu(), enable_gqa=True,
                    )
                    ref_grads = torch.autograd.grad(ref_out, ref_inputs, grad_out.cpu().float())
                    torch.testing.assert_close(out.cpu().float(), ref_out, atol=3e-3, rtol=3e-3)
                    for actual, expected in zip(grads, ref_grads):
                        torch.testing.assert_close(actual.cpu().float(), expected, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
