# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.attention.flex_attention 接口功能正确性
API 名称：torch.nn.attention.flex_attention.flex_attention
API 签名：flex_attention(query, key, value, score_mod=None, block_mask=None, scale=None,
enable_gqa=False, return_lse=False, kernel_options=None, *, return_aux=None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | block_mask=None 默认全注意力与 create_block_mask 因果掩码两分支 | 已覆盖（flex_attention_fp32_full_attention_vs_cpu/
flex_attention_block_mask_causal_vs_reference） |
| 枚举选项 | BACKEND 枚举：AUTO 合法值与非法值、与 FORCE_USE_FLEX_ATTENTION 互斥 | 已覆盖（
flex_attention_kernel_options_backend_auto_vs_cpu/flex_attention_invalid_backend_option_raises/
flex_attention_backend_with_legacy_force_raises） |
| 参数类型 | scale float、score_mod 可调用、kernel_options dict、return_aux AuxRequest | 已覆盖（flex_attention_custom_scale_vs_cpu/
flex_attention_score_mod_causal_vs_reference/
flex_attention_reserved_kernel_option_raises/
flex_attention_return_aux_lse_vs_reference） |
| 传参与不传参 | 默认 scale/默认全注意力与显式 scale/score_mod/block_mask/enable_gqa | 已覆盖（flex_attention_fp32_full_attention_vs_cpu/
flex_attention_custom_scale_vs_cpu/flex_attention_gqa_vs_cpu） |
| 等价类/边界值 | fp32/fp16/bfloat16 dtype、嵌入维不匹配、批不匹配、3D 输入、嵌套 Tensor | 已覆盖（flex_attention_fp16_vs_cpu/flex_attention_bfloat16_vs_cpu/
flex_attention_embed_dim_mismatch_raises/flex_attention_batch_mismatch_raises/
flex_attention_3d_input_raises/flex_attention_nested_tensor_raises） |
| 正常传参场景 | 输出/lse/max_scores/反向梯度与 CPU Golden 或手工参考一致；重复调用结果一致 | 已覆盖（flex_attention_fp32_full_attention_vs_cpu/
flex_attention_return_aux_lse_vs_reference/
flex_attention_return_aux_max_scores_vs_reference/
flex_attention_backward_grad_vs_reference/
flex_attention_repeated_calls_deterministic） |
| 异常传参场景 | 头数不匹配、保留 kernel_options 键、return_lse 与 return_aux 冲突、混合设备输入 | 已覆盖（flex_attention_head_mismatch_raises/
flex_attention_reserved_kernel_option_raises/
flex_attention_return_lse_with_return_aux_raises/
flex_attention_mixed_device_input_raises） |
| 混合设备类型 | q/k/v 混合设备（设备/CPU）输入被设备校验拒绝并抛 ValueError | 已覆盖（flex_attention_mixed_device_input_raises） |

未覆盖项及原因：
- BACKEND='TRITON'/'TRITON_DECODE'/'FLASH' 枚举值仅在对应后端可用的平台有意义（FLASH 需
cute-dSL flash 包），本文件仅覆盖 AUTO 与非法值，不将其他平台的后端可用性固化为断言。
- 序列化、Hook：flex_attention 为无状态函数式入口，无序列化/Hook 契约，不适用。
- 设备门控：用例仅在 NPU 或 CUDA 可用时执行；两者都不可用时 DEVICE_NAME 置 None，整类在
类初始化前显式跳过（skipIf，基类 setUpClass 会先于 per-test setUp 选卡），不回退 cpu，避免
数值对照退化为 CPU 自比对、混合设备断言失去设备差异。
"""
import math
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement tests actually run. Plain CI
# collection skips the class before any torch_npu import or device init.
# When neither NPU nor CUDA is available, DEVICE_NAME stays None: the device
# cases are then skipped explicitly in setUp instead of silently degrading to
# CPU-only self-comparisons or losing the mixed-device contract.
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else None

DEVICE_SKIP_REASON = (
    "no NPU/CUDA device available; these supplement cases require an accelerator"
)

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


import torch.nn.attention.flex_attention as fa  # noqa: E402


B, H, M, D = 2, 2, 16, 8


def _causal_score_mod(score, b, h, q_idx, kv_idx):
    """Mask future positions through score_mod (-inf outside the causal band)."""
    return torch.where(q_idx >= kv_idx, score, -float('inf'))


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _reference_attention(q_cpu, k_cpu, v_cpu, scale, score_mod=None):
    """Dense softmax attention reference in fp32 (score_mod applied in-place)."""
    scores = (q_cpu.float() @ k_cpu.float().transpose(-2, -1)) * scale
    if score_mod is not None:
        b_idx = torch.arange(scores.size(0)).view(-1, 1, 1, 1)
        h_idx = torch.arange(scores.size(1)).view(1, -1, 1, 1)
        m_idx = torch.arange(scores.size(2)).view(1, 1, -1, 1)
        n_idx = torch.arange(scores.size(3)).view(1, 1, 1, -1)
        scores = score_mod(scores, b_idx, h_idx, m_idx, n_idx)
    return scores.softmax(-1) @ v_cpu.float(), scores


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
# Class-level guard: when torch_npu is installed but no accelerator exists,
# the torch_npu TestCase base class calls torch.npu.set_device in setUpClass
# before any per-test setUp could skip, so the whole class must be skipped
# before base-class initialization runs.
@unittest.skipIf(DEVICE_NAME is None, DEVICE_SKIP_REASON)
class TestFlexAttentionFlexAttention(TestCase):
    """Test cases for torch.nn.attention.flex_attention.flex_attention (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        if self.device_name is None:
            # Without an accelerator there is no device path to exercise: skip
            # rather than run on CPU, where mixed-device inputs would not differ
            # and value checks would only self-compare.
            self.skipTest(DEVICE_SKIP_REASON)
        self.device = torch.device(self.device_name)

    def _make_qkv(self, batch=B, heads=H, seq=M, dim=D, dtype=torch.float32, seed=0):
        """Create identical q/k/v on CPU and the target device from one seed."""
        gen = torch.Generator(device='cpu').manual_seed(seed)
        tensors = []
        for _ in range(3):
            cpu_tensor = torch.randn(batch, heads, seq, dim, generator=gen, dtype=dtype)
            tensors.append((cpu_tensor, cpu_tensor.to(self.device)))
        return tensors

    # ------------------------------------------------------------------
    # Functional golden comparisons
    # ------------------------------------------------------------------

    def test_flex_attention_fp32_full_attention_vs_cpu(self):
        """fp32 full attention matches the CPU flex golden."""
        qkv = self._make_qkv()
        out = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1])
        cpu_out = fa.flex_attention(qkv[0][0], qkv[1][0], qkv[2][0])
        self.assertEqual(out.shape, (B, H, M, D))
        torch.testing.assert_close(out.cpu(), cpu_out)

    def test_flex_attention_fp16_vs_cpu(self):
        """fp16 inputs match the CPU flex golden within fp16 tolerance."""
        qkv = self._make_qkv(dtype=torch.float16, seed=1)
        out = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1])
        cpu_out = fa.flex_attention(qkv[0][0], qkv[1][0], qkv[2][0])
        torch.testing.assert_close(out.cpu(), cpu_out, atol=2e-3, rtol=2e-3)

    def test_flex_attention_bfloat16_vs_cpu(self):
        """bfloat16 inputs match the CPU flex golden within bf16 tolerance."""
        qkv = self._make_qkv(dtype=torch.bfloat16, seed=2)
        out = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1])
        cpu_out = fa.flex_attention(qkv[0][0], qkv[1][0], qkv[2][0])
        torch.testing.assert_close(out.cpu(), cpu_out, atol=2e-2, rtol=2e-2)

    def test_flex_attention_custom_scale_vs_cpu(self):
        """Explicit scale=0.5 matches the CPU flex golden."""
        qkv = self._make_qkv(seed=3)
        out = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1], scale=0.5)
        cpu_out = fa.flex_attention(qkv[0][0], qkv[1][0], qkv[2][0], scale=0.5)
        torch.testing.assert_close(out.cpu(), cpu_out)
        ref, _ = _reference_attention(qkv[0][0], qkv[1][0], qkv[2][0], 0.5)
        torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_flex_attention_score_mod_causal_vs_reference(self):
        """score_mod causal masking matches the dense masked reference."""
        qkv = self._make_qkv(seed=4)
        out = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1], score_mod=_causal_score_mod)
        ref, _ = _reference_attention(
            qkv[0][0], qkv[1][0], qkv[2][0], 1.0 / math.sqrt(D), score_mod=_causal_score_mod
        )
        torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_flex_attention_block_mask_causal_vs_reference(self):
        """create_block_mask causal mask matches the dense masked reference."""
        qkv = self._make_qkv(seed=5)
        block_mask = fa.create_block_mask(
            _causal_mask_mod, B, H, M, M, device=qkv[0][1].device
        )
        out = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1], block_mask=block_mask)
        _, scores = _reference_attention(qkv[0][0], qkv[1][0], qkv[2][0], 1.0 / math.sqrt(D))
        causal_scores = scores.masked_fill(
            ~torch.tril(torch.ones(M, M, dtype=torch.bool)), -float('inf')
        )
        causal_ref = causal_scores.softmax(-1) @ qkv[2][0].float()
        torch.testing.assert_close(out.cpu(), causal_ref, atol=1e-5, rtol=1e-5)

    def test_flex_attention_gqa_vs_cpu(self):
        """enable_gqa with Hq=4/Hkv=2 matches the CPU flex golden."""
        batch, seq, dim = B, M, D
        gen = torch.Generator(device='cpu').manual_seed(6)
        q_cpu = torch.randn(batch, 4, seq, dim, generator=gen)
        k_cpu = torch.randn(batch, 2, seq, dim, generator=gen)
        v_cpu = torch.randn(batch, 2, seq, dim, generator=gen)
        out = fa.flex_attention(q_cpu.to(self.device), k_cpu.to(self.device),
                                v_cpu.to(self.device), enable_gqa=True)
        cpu_out = fa.flex_attention(q_cpu, k_cpu, v_cpu, enable_gqa=True)
        self.assertEqual(out.shape, (batch, 4, seq, dim))
        torch.testing.assert_close(out.cpu(), cpu_out)
        # Cross-check against the repeat_interleave (blocked) head mapping.
        kv_heads = k_cpu.repeat_interleave(2, dim=1)
        v_heads = v_cpu.repeat_interleave(2, dim=1)
        ref, _ = _reference_attention(q_cpu, kv_heads, v_heads, 1.0 / math.sqrt(dim))
        torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-5)

    def test_flex_attention_kernel_options_backend_auto_vs_cpu(self):
        """kernel_options BACKEND='AUTO' matches the CPU flex golden."""
        qkv = self._make_qkv(seed=7)
        out = fa.flex_attention(
            qkv[0][1], qkv[1][1], qkv[2][1], kernel_options={'BACKEND': 'AUTO'}
        )
        cpu_out = fa.flex_attention(
            qkv[0][0], qkv[1][0], qkv[2][0], kernel_options={'BACKEND': 'AUTO'}
        )
        torch.testing.assert_close(out.cpu(), cpu_out)

    def test_flex_attention_return_aux_lse_vs_reference(self):
        """return_aux lse matches the manual logsumexp reference."""
        qkv = self._make_qkv(seed=8)
        out, aux = fa.flex_attention(
            qkv[0][1], qkv[1][1], qkv[2][1], return_aux=fa.AuxRequest(lse=True)
        )
        scale = 1.0 / math.sqrt(D)
        ref, scores = _reference_attention(qkv[0][0], qkv[1][0], qkv[2][0], scale)
        torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-5)
        self.assertIsNotNone(aux.lse)
        self.assertIsNone(aux.max_scores)
        torch.testing.assert_close(aux.lse.cpu(), scores.logsumexp(dim=-1))

    def test_flex_attention_return_aux_max_scores_vs_reference(self):
        """return_aux max_scores matches the manual row-max reference."""
        qkv = self._make_qkv(seed=9)
        out, aux = fa.flex_attention(
            qkv[0][1], qkv[1][1], qkv[2][1], return_aux=fa.AuxRequest(max_scores=True)
        )
        scale = 1.0 / math.sqrt(D)
        ref, scores = _reference_attention(qkv[0][0], qkv[1][0], qkv[2][0], scale)
        torch.testing.assert_close(out.cpu(), ref, atol=1e-5, rtol=1e-5)
        self.assertIsNone(aux.lse)
        self.assertIsNotNone(aux.max_scores)
        torch.testing.assert_close(aux.max_scores.cpu(), scores.max(dim=-1).values)

    def test_flex_attention_backward_grad_vs_reference(self):
        """query gradient matches a float64 autograd math reference."""
        gen = torch.Generator(device='cpu').manual_seed(10)
        q_cpu = torch.randn(B, H, M, D, generator=gen, dtype=torch.float64)
        k_cpu = torch.randn(B, H, M, D, generator=gen, dtype=torch.float64)
        v_cpu = torch.randn(B, H, M, D, generator=gen, dtype=torch.float64)
        q_ref = q_cpu.clone().requires_grad_(True)
        scores = (q_ref @ k_cpu.transpose(-2, -1)) / math.sqrt(D)
        (scores.softmax(-1) @ v_cpu).sum().backward()
        q_npu = q_cpu.float().to(self.device).requires_grad_(True)
        out = fa.flex_attention(
            q_npu, k_cpu.float().to(self.device), v_cpu.float().to(self.device)
        )
        out.sum().backward()
        self.assertIsNotNone(q_npu.grad)
        torch.testing.assert_close(q_npu.grad.cpu().double(), q_ref.grad, atol=1e-5, rtol=1e-5)

    def test_flex_attention_repeated_calls_deterministic(self):
        """Repeated calls on identical inputs return identical results."""
        qkv = self._make_qkv(seed=11)
        first = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1])
        second = fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1])
        self.assertTrue(torch.equal(first, second))

    # ------------------------------------------------------------------
    # Validation and exception contracts (device independent)
    # ------------------------------------------------------------------

    def test_flex_attention_embed_dim_mismatch_raises(self):
        """Mismatched query/key embedding dims raise ValueError."""
        qkv = self._make_qkv(seed=12)
        k_bad = torch.randn(B, H, M, 16, generator=torch.Generator().manual_seed(0))
        with self.assertRaises(ValueError):
            fa.flex_attention(qkv[0][1], k_bad.to(self.device), qkv[2][1])

    def test_flex_attention_head_mismatch_raises(self):
        """Head-count mismatch without GQA raises ValueError."""
        qkv = self._make_qkv(seed=13)
        k_bad = torch.randn(B, 3, M, D, generator=torch.Generator().manual_seed(1))
        v_bad = torch.randn(B, 3, M, D, generator=torch.Generator().manual_seed(2))
        with self.assertRaises(ValueError):
            fa.flex_attention(qkv[0][1], k_bad.to(self.device), v_bad.to(self.device))

    def test_flex_attention_batch_mismatch_raises(self):
        """Batch mismatch with no block_mask raises ValueError."""
        qkv = self._make_qkv(seed=14)
        k_other = torch.randn(1, H, M, D, generator=torch.Generator().manual_seed(3))
        v_other = torch.randn(1, H, M, D, generator=torch.Generator().manual_seed(4))
        with self.assertRaises(ValueError):
            fa.flex_attention(qkv[0][1], k_other.to(self.device), v_other.to(self.device))

    def test_flex_attention_nested_tensor_raises(self):
        """NestedTensor inputs raise NotImplementedError."""
        qkv = self._make_qkv(seed=15)
        # A list of 2D tensors yields a 3D nested tensor; pass it directly so
        # the nested-input rejection is deterministic on every device (calling
        # unsqueeze(0) on a nested tensor is itself unsupported on CUDA).
        nested = torch.nested.nested_tensor(
            [torch.randn(M, D), torch.randn(M, D)],
            device=qkv[0][1].device,
        )
        with self.assertRaises(NotImplementedError):
            fa.flex_attention(nested, qkv[1][1], qkv[2][1])

    def test_flex_attention_3d_input_raises(self):
        """Non-4D inputs raise NotImplementedError (NYI)."""
        qkv = self._make_qkv(seed=16)
        with self.assertRaises(NotImplementedError):
            fa.flex_attention(qkv[0][1][0], qkv[1][1][0], qkv[2][1][0])

    def test_flex_attention_reserved_kernel_option_raises(self):
        """OUTPUT_LOGSUMEXP is reserved and raises AssertionError."""
        qkv = self._make_qkv(seed=17)
        with self.assertRaises(AssertionError):
            fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1],
                              kernel_options={'OUTPUT_LOGSUMEXP': True})

    def test_flex_attention_invalid_backend_option_raises(self):
        """Unknown BACKEND value raises ValueError."""
        qkv = self._make_qkv(seed=18)
        with self.assertRaises(ValueError):
            fa.flex_attention(qkv[0][1], qkv[1][1], qkv[2][1],
                              kernel_options={'BACKEND': 'FOO'})

    def test_flex_attention_backend_with_legacy_force_raises(self):
        """BACKEND combined with FORCE_USE_FLEX_ATTENTION raises RuntimeError."""
        qkv = self._make_qkv(seed=19)
        with self.assertRaises(RuntimeError):
            fa.flex_attention(
                qkv[0][1], qkv[1][1], qkv[2][1],
                kernel_options={'BACKEND': 'AUTO', 'FORCE_USE_FLEX_ATTENTION': True},
            )

    def test_flex_attention_return_lse_with_return_aux_raises(self):
        """return_lse and return_aux are mutually exclusive."""
        qkv = self._make_qkv(seed=20)
        with self.assertRaises(ValueError):
            fa.flex_attention(
                qkv[0][1], qkv[1][1], qkv[2][1],
                return_lse=True, return_aux=fa.AuxRequest(lse=True),
            )

    def test_flex_attention_mixed_device_input_raises(self):
        """Mixed-device q/k/v inputs are rejected with ValueError."""
        qkv = self._make_qkv(seed=21)
        with self.assertRaises(ValueError):
            fa.flex_attention(qkv[0][1], qkv[1][0], qkv[2][0])


if __name__ == "__main__":
    run_tests()
