import unittest

import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code

import torch_npu  # noqa: F401


class TestPersistentSoftmaxPadding(unittest.TestCase):
    @unittest.skipUnless(torch.npu.is_available(), "requires NPU")
    def test_dynamic_masked_softmax_tail(self):
        from torch_npu._inductor import config as npu_config

        if not npu_config.is_ascend950:
            self.skipTest("requires Ascend 950 template kernels")

        def attention(query, key, mask, value):
            batch, width = mask.shape
            scores = torch.bmm(query, key).reshape(batch, 4, 3, width)
            full = torch.full((batch, 3), 1, device=query.device, dtype=query.dtype)
            weights = torch.bmm(full.unsqueeze(2), mask.unsqueeze(1)).unsqueeze(1)
            logits = (scores * 0.125 + (1 - weights) * -10000).float()
            exp = (logits - logits.amax(dim=-1, keepdim=True)).exp()
            probabilities = (exp / exp.sum(dim=-1, keepdim=True) * weights).half()
            return torch.bmm(probabilities.reshape(4 * batch, 3, width), value)

        with config.patch(max_autotune=True, compile_threads=1, cpp_wrapper=False):
            compiled = torch.compile(attention, fullgraph=True)
            for seed in (0, 1, 2):
                generator = torch.Generator().manual_seed(seed)
                query = torch.randn(1024, 3, 64, generator=generator).half().npu()
                key = torch.randn(1024, 64, 229, generator=generator).half().npu()
                lengths = torch.randint(31, 230, (256,), generator=generator)
                mask = (torch.arange(229)[None, :] < lengths[:, None]).half().npu()
                value = torch.randn(1024, 229, 64, generator=generator).half().npu()
                for tensor, axes in (
                    (query, (0,)), (key, (0, 2)), (mask, (0, 1)), (value, (0, 1))
                ):
                    for axis in axes:
                        if tensor is mask and axis == 1:
                            torch._dynamo.mark_dynamic(tensor, axis, min=31, max=229)
                        else:
                            torch._dynamo.mark_dynamic(tensor, axis)
                args = query, key, mask, value
                expected = attention(*args)
                actual, sources = run_and_get_code(compiled, *args)
                torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0)
                if seed == 0:
                    source = "\n".join(sources)
                    self.assertIn("@triton_heuristics.persistent_reduction(", source)
                    self.assertRegex(source, r"tl\.load\([^\n]*r\d+_mask")
                    self.assertRegex(source, r"tl\.store\([^\n]*r\d+_mask")
                    self.assertRegex(source, r"tl\.where\([^\n]*r\d+_mask")
                first = actual.clone()
                for _ in range(4):
                    actual = compiled(*args)
                    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0)
                    torch.testing.assert_close(actual, first, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
