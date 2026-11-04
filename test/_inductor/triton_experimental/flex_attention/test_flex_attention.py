import math
import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import TestCase, run_tests

try:
    import torch_npu  # noqa: F401
    import torch_npu._inductor  # noqa: F401
except ImportError:
    torch_npu = None

KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64}
_NPU_AVAILABLE = bool(hasattr(torch, "npu") and torch.npu.is_available())


class TestExperimentalFlexAttentionSource(TestCase):
    def test_default_components_are_reused(self):
        if torch_npu is None:
            self.skipTest("requires torch_npu")
        from torch_npu._inductor.kernel import flex_attention as default_lowering
        from torch_npu._inductor.kernel import flexattention_template as default
        from torch._inductor.kernel.flex.common import load_flex_template
        from torch_npu._inductor.triton_experimental.flex_attention import flex_attention_template as te
        from torch_npu._inductor.triton_experimental.flex_attention.flex_attention_adapter import _compile_copy
        self.assertIs(te.load_flex_template, load_flex_template)
        self.assertIs(te.compute_sparse_mask_kernel_compact, default.compute_sparse_mask_kernel_compact)
        self.assertIs(te.compact_sparse_mask_grid, default.compact_sparse_mask_grid)
        names = [v.name for n, v in vars(te).items() if n.startswith("flex_") and hasattr(v, "name")]
        self.assertTrue(names and all(name.startswith("te_") for name in names))
        self.assertIsNot(_compile_copy(default_lowering._register_npu_inductor_flex_attention, {}, transform=True),
                         default_lowering._register_npu_inductor_flex_attention)

    def test_fusion_budget_boundaries(self):
        if torch_npu is None:
            self.skipTest("requires torch_npu")
        from torch_npu._inductor.triton_experimental.flex_attention.flex_attention import (
            _LINEARIZE_FUSION_CAPTURE_BUDGET as captures,
            _LINEARIZE_FUSION_NODE_BUDGET as nodes,
            _prepare_linearize_fusion_options as prepare,
        )

        def graph(count):
            node = type("N", (), {"op": "call_function"})
            return type("Graph", (), {"graph": type("FX", (), {"nodes": [node() for _ in range(count)]})()})()
        within = prepare({"FUSED_CAPTURED_BUFFERS": [None] * captures}, graph(nodes), graph(0))
        self.assertTrue(within["FUSE_SMALL_SUBGRAPHS"])
        self.assertFalse(prepare({}, graph(nodes + 1), graph(0))["FUSE_SMALL_SUBGRAPHS"])
        over = {"FUSED_CAPTURED_BUFFERS": [None] * (captures + 1)}
        self.assertFalse(prepare(over, graph(0), graph(0))["FUSE_SMALL_SUBGRAPHS"])


def _causal(_batch, _head, q, k):
    return q >= k


def _window(_batch, _head, q, k):
    return (q >= k) & (q - k <= 16)


def _half_score(score, _batch, _head, _q, _k):
    return score * 0.5


def _reference(q, k, v, *, scale=1.0, window=None):
    scores = q.float() @ k.float().transpose(-2, -1) * (scale / math.sqrt(q.shape[-1]))
    qi = torch.arange(q.shape[-2], device=q.device)[:, None]
    ki = torch.arange(k.shape[-2], device=q.device)[None, :]
    mask = qi >= ki
    if window is not None:
        mask &= qi - ki <= window
    return (torch.softmax(scores.masked_fill(~mask, -float("inf")), -1) @ v.float()).to(q.dtype)


@unittest.skipUnless(_NPU_AVAILABLE, "requires an NPU")
class TestFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch.manual_seed(2026)
        self.shape = (1, 2, 64, 32)

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _inputs(self, grad=False):
        return [torch.randn(self.shape, device="npu", dtype=torch.bfloat16, requires_grad=grad) for _ in range(3)]

    def _mask(self, fn):
        return create_block_mask(fn, B=self.shape[0], H=self.shape[1], Q_LEN=self.shape[2],
                                 KV_LEN=self.shape[2], device="npu")

    def _compile(self, fn):
        return torch.compile(fn, backend="inductor", fullgraph=True,
                             options={"npu_backend": "triton_experimental"})

    def _check_code(self, codes, backward=False):
        code = "\n".join(codes)
        self.assertIn("from torch_npu._inductor.triton_experimental import npu_triton_heuristics", code)
        self.assertNotIn("from torch_npu._inductor.runtime.triton_heuristics", code)
        self.assertIn("Original ATen: [flex_attention]", code)
        if backward:
            self.assertIn("flex_attention_backward", code)

    def _forward(self, mask, *, score_mod=None, reference=None):
        inputs = self._inputs()
        block_mask = self._mask(mask)

        def fn(q, k, v):
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask,
                                  kernel_options=KERNEL_OPTIONS)
        actual, codes = run_and_get_code(self._compile(fn), *inputs)
        torch.testing.assert_close(actual, reference(*inputs) if reference else fn(*inputs), atol=0.04, rtol=0.04)
        self._check_code(codes)

    def test_causal_forward(self):
        self._forward(_causal, reference=_reference)

    def test_captured_tensor_index_in_mask(self):
        document_id = torch.arange(self.shape[-2], device="npu", dtype=torch.int32) // 16
        self._forward(lambda _b, _h, q, k: document_id[q] == document_id[k])

    def test_query_offset_index_in_mask(self):
        query_offsets = torch.arange(self.shape[-2], device="npu", dtype=torch.int32) // 16 * 16
        self._forward(lambda _b, _h, q, k: query_offsets[q] <= k)

    def test_score_mod_and_sliding_window_mask(self):
        self._forward(_window, score_mod=_half_score,
                      reference=lambda q, k, v: _reference(q, k, v, scale=0.5, window=16))

    def test_causal_forward_backward(self):
        inputs = self._inputs(grad=True)
        block_mask = self._mask(_causal)

        def attention_loss(q, k, v):
            output = flex_attention(q, k, v, block_mask=block_mask, kernel_options=KERNEL_OPTIONS)
            return output.float().square().mean(), output
        compiled = self._compile(attention_loss)

        def run(q, k, v):
            loss, output = compiled(q, k, v)
            return output, torch.autograd.grad(loss, (q, k, v))
        (actual, gradients), codes = run_and_get_code(run, *inputs)
        reference_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        expected = _reference(*reference_inputs)
        expected_grads = torch.autograd.grad(expected.float().square().mean(), reference_inputs)
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
        for value, target in zip(gradients, expected_grads):
            torch.testing.assert_close(value, target, atol=0.12, rtol=0.12)
        self._check_code(codes, backward=True)

    def test_causal_backward_unreachable_kv_tail(self):
        q = torch.randn((1, 1, 384, 32), device="npu", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn((1, 1, 512, 32), device="npu", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        block_mask = create_block_mask(_causal, B=1, H=1, Q_LEN=384, KV_LEN=512, device="npu")

        def attention(q, k, v):
            return flex_attention(q, k, v, block_mask=block_mask, kernel_options=KERNEL_OPTIONS)
        compiled = self._compile(attention)
        reference = _reference(q, k, v)
        reference_grads = torch.autograd.grad(reference.float().square().mean(), (q, k, v))
        for _ in range(3):
            output = compiled(q, k, v)
            grads = torch.autograd.grad(output.float().square().mean(), (q, k, v))
            torch.testing.assert_close(output, reference, atol=0.04, rtol=0.04)
            for actual, expected in zip(grads, reference_grads):
                torch.testing.assert_close(actual, expected, atol=0.12, rtol=0.12)
            for gradient in grads[1:]:
                self.assertEqual(torch.count_nonzero(gradient[..., 384:, :]).item(), 0)


if __name__ == "__main__":
    run_tests()
