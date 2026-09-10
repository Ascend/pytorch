import functools
import math
import re
from unittest.mock import patch

import torch
import torch_npu
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor import config as npu_config
from torch._inductor import metrics
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing import FileCheck
from torch_npu.testing.testcase import TestCase, run_tests


def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _diagonal_mask(b, h, q_idx, kv_idx):
    return q_idx == kv_idx


def _full_mask(b, h, q_idx, kv_idx):
    return q_idx >= 0


class TestFlexAttention(TestCase):
    def setUp(self):
        super().setUp()
        budget_patch = patch.object(
            npu_config.flex_attention, "fwd_mask_workspace_bytes", 256 << 20
        )
        budget_patch.start()
        self.addCleanup(budget_patch.stop)
        torch._dynamo.reset()
        metrics.reset()

    def _make_block_mask(self, mask_mod, batch, q_len, kv_len):
        return create_block_mask(
            mask_mod,
            batch,
            1,
            q_len,
            kv_len,
            device="npu",
            BLOCK_SIZE=(128, 128),
        )

    def _make_inputs(
        self,
        *,
        q_batch=1,
        kv_batch=1,
        q_heads=2,
        kv_heads=2,
        q_len=256,
        kv_len=256,
        head_dim=64,
        dtype=torch.float16,
        requires_grad=False,
    ):
        def make(shape):
            return torch.randn(
                shape,
                device="npu",
                dtype=dtype,
                requires_grad=requires_grad,
            )

        return (
            make((q_batch, q_heads, q_len, head_dim)),
            make((kv_batch, kv_heads, kv_len, head_dim)),
            make((kv_batch, kv_heads, kv_len, head_dim)),
        )

    def _dense_mask(self, mask_kind, q_len, kv_len):
        q_idx = torch.arange(q_len, device="npu")[:, None]
        kv_idx = torch.arange(kv_len, device="npu")[None, :]
        if mask_kind == "causal":
            return q_idx >= kv_idx
        if mask_kind == "diagonal":
            return q_idx == kv_idx
        if mask_kind == "full":
            return torch.ones((q_len, kv_len), device="npu", dtype=torch.bool)
        raise AssertionError(f"unknown mask kind: {mask_kind}")

    def _sdpa_reference(self, q, k, v, mask_kind, enable_gqa=False):
        dense_mask = self._dense_mask(mask_kind, q.shape[-2], k.shape[-2])
        if k.shape[0] == 1 and q.shape[0] != 1:
            k = k.expand(q.shape[0], -1, -1, -1)
            v = v.expand(q.shape[0], -1, -1, -1)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=dense_mask[None, None],
            enable_gqa=enable_gqa,
        )

    def _assert_close(self, actual, expected, dtype, *, gradients=False):
        if dtype == torch.bfloat16:
            atol = rtol = 8e-2 if gradients else 2e-2
        elif gradients:
            atol = rtol = 2e-2
        else:
            atol = rtol = 1e-2
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.isfinite(expected).all().item())
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )

    def _assert_mask_out_codegen(self, code, *, workspace):
        source = "\n".join(code)
        if workspace:
            expected = "triton_flex_attention_fwd_workspace_mask_compact"
            unexpected = "triton_flex_attention_fwd_mask_compact"
        else:
            expected = "triton_flex_attention_fwd_mask_compact"
            unexpected = "triton_flex_attention_fwd_workspace_mask_compact"
        self.assertRegex(source, rf"{expected}_[0-9]+\s*=")
        self.assertNotIn(unexpected, source)
        self.assertIsNotNone(
            re.search(
                r"triton_flex_attention_fwd_mask_out_[0-9]+\s*=",
                source,
            )
        )
        self.assertIn("arg_SPARSE_MASK", source)
        self.assertIsNone(
            re.search(
                r"triton_flex_attention_fwd_mask_out_mojo_[0-9]+\s*=",
                source,
            )
        )

    def _run_compile_vs_sdpa(
        self,
        *,
        mask_mod,
        mask_kind,
        q_batch=1,
        kv_batch=1,
        q_heads=2,
        kv_heads=2,
        q_len=256,
        kv_len=256,
        head_dim=64,
        dtype=torch.float16,
        enable_gqa=False,
    ):
        q, k, v = self._make_inputs(
            q_batch=q_batch,
            kv_batch=kv_batch,
            q_heads=q_heads,
            kv_heads=kv_heads,
            q_len=q_len,
            kv_len=kv_len,
            head_dim=head_dim,
            dtype=dtype,
        )
        block_mask = self._make_block_mask(
            mask_mod, q_batch, q_len, kv_len
        )
        compiled = torch.compile(
            flex_attention, fullgraph=True, dynamic=False
        )
        actual, code = run_and_get_code(
            compiled,
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=enable_gqa,
        )
        expected = self._sdpa_reference(
            q, k, v, mask_kind, enable_gqa=enable_gqa
        )
        self._assert_close(actual, expected, dtype)
        self._assert_mask_out_codegen(code, workspace=True)
        return block_mask

    def test_epilogue_fused(self):
        @torch.compile
        def f(q, k, v):
            return flex_attention(q, k, v).cos()

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device="npu") for _ in range(3)
        )
        _, code = run_and_get_code(f, q, k, v)

        # FileCheck().check("triton_tem_fused").check_not("poi_fused_cos").run(
        #     code[0]
        # )
        accessed_bytes = 1 * 8 * 1024 * 64 * torch.float32.itemsize
        num_accesses = 6
        # TODO: Get rid of this fudge factor
        # We need this fudge factor for now as we write the extraneous logsumexp
        num_accesses += 1
        self.assertLess(metrics.num_bytes_accessed, accessed_bytes * num_accesses)

    def test_kernel_options_argument_is_respected(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device="npu",
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        _, code = run_and_get_code(
            torch.compile(flex_attention),
            q,
            k,
            v,
            kernel_options={"BLOCK_M": 16},
        )

        FileCheck().check("BLOCK_M : tl.constexpr = 16").run(code[0])

    def test_compile_vs_sdpa_mask_block_kinds(self):
        cases = (
            ("all_sparse", _diagonal_mask, "diagonal"),
            ("full", _full_mask, "full"),
            ("mixed", _causal_mask, "causal"),
        )
        for name, mask_mod, mask_kind in cases:
            with self.subTest(name=name):
                block_mask = self._run_compile_vs_sdpa(
                    mask_mod=mask_mod,
                    mask_kind=mask_kind,
                )
                sparse = block_mask.kv_num_blocks
                full = block_mask.full_kv_num_blocks
                if name == "all_sparse":
                    self.assertTrue((sparse > 0).any().item())
                    self.assertTrue((full == 0).all().item())
                elif name == "full":
                    self.assertTrue((sparse == 0).all().item())
                    self.assertTrue((full > 0).any().item())
                else:
                    self.assertTrue((sparse > 0).any().item())
                    self.assertTrue((full > 0).any().item())

    def test_compile_vs_sdpa_non_divisible_lengths(self):
        self._run_compile_vs_sdpa(
            mask_mod=_causal_mask,
            mask_kind="causal",
            q_len=257,
            kv_len=385,
        )

    def test_compile_vs_sdpa_batch_broadcast_gqa(self):
        self._run_compile_vs_sdpa(
            mask_mod=_causal_mask,
            mask_kind="causal",
            q_batch=2,
            kv_batch=1,
            q_heads=4,
            kv_heads=2,
            enable_gqa=True,
        )

    def test_compile_vs_sdpa_dynamic_shapes(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        backend = CompileCounterWithBackend("inductor")
        compiled = torch.compile(
            flex_attention,
            backend=backend,
            fullgraph=True,
            dynamic=True,
        )
        code = None
        for index, (q_len, kv_len) in enumerate(((129, 257), (193, 321))):
            with self.subTest(q_len=q_len, kv_len=kv_len):
                q, k, v = self._make_inputs(q_len=q_len, kv_len=kv_len)
                block_mask = self._make_block_mask(
                    _causal_mask, 1, q_len, kv_len
                )
                if index == 0:
                    actual, code = run_and_get_code(
                        compiled, q, k, v, block_mask=block_mask
                    )
                else:
                    actual = compiled(q, k, v, block_mask=block_mask)
                expected = self._sdpa_reference(q, k, v, "causal")
                self._assert_close(actual, expected, q.dtype)
        self.assertEqual(backend.frame_count, 1)
        self.assertIsNotNone(code)
        self._assert_mask_out_codegen(code, workspace=False)

    def test_compile_vs_sdpa_dtypes_and_head_dims(self):
        for dtype, head_dim in (
            (torch.bfloat16, 64),
            (torch.float16, 128),
        ):
            with self.subTest(dtype=dtype, head_dim=head_dim):
                self._run_compile_vs_sdpa(
                    mask_mod=_causal_mask,
                    mask_kind="causal",
                    head_dim=head_dim,
                    dtype=dtype,
                )

    def test_compile_vs_sdpa_output_lse_and_gradients(self):
        dtype = torch.float16
        q, k, v = self._make_inputs(
            q_len=257,
            kv_len=385,
            dtype=dtype,
            requires_grad=True,
        )
        q_ref, k_ref, v_ref = (
            tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v)
        )
        block_mask = self._make_block_mask(_causal_mask, 1, 257, 385)
        compiled = torch.compile(
            flex_attention, fullgraph=True, dynamic=False
        )
        (actual, actual_lse), code = run_and_get_code(
            compiled,
            q,
            k,
            v,
            block_mask=block_mask,
            return_lse=True,
        )
        expected = self._sdpa_reference(q_ref, k_ref, v_ref, "causal")

        dense_mask = self._dense_mask("causal", 257, 385)
        scores = torch.matmul(q_ref.float(), k_ref.float().transpose(-2, -1))
        scores = scores * (1.0 / math.sqrt(q_ref.shape[-1]))
        scores = scores.masked_fill(~dense_mask[None, None], -float("inf"))
        expected_lse = torch.logsumexp(scores, dim=-1)

        self._assert_close(actual, expected, dtype)
        self._assert_close(actual_lse, expected_lse, dtype)
        self._assert_mask_out_codegen(code, workspace=True)

        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
        expected_grads = torch.autograd.grad(
            expected, (q_ref, k_ref, v_ref), grad
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self._assert_close(
                actual_grad, expected_grad, dtype, gradients=True
            )


if __name__ == "__main__":
    run_tests()
