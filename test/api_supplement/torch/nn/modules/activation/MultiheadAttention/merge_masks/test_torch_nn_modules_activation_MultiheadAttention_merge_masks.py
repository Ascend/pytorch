# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.modules.activation.MultiheadAttention.merge_masks 接口功能正确性
API 名称：torch.nn.modules.activation.MultiheadAttention.merge_masks
API 签名：merge_masks(self, attn_mask, key_padding_mask, query) -> (Tensor | None, int | None)

覆盖维度表：
| 覆盖维度         | 说明                                              | 覆盖情况                       |
|------------------|---------------------------------------------------|--------------------------------|
| 空/非空          | attn_mask/key_padding_mask 为 None 与合法张量     | 已覆盖（四组合全覆盖）         |
| 参数类型         | attn_mask 2D bool/float、3D bool；kpm 2D bool     | 已覆盖                         |
| 传参与不传参     | 单掩码返回原引用；双掩码合并加和                  | 已覆盖                         |
| 等价类/边界值    | 无掩码返回 (None, None)                           | 已覆盖                         |
| 正常传参场景     | 形状展开、mask_type 语义、加和数值                | 已覆盖                         |
| 异常传参场景     | attn_mask 序列长度与 query 不匹配报 RuntimeError  | 已覆盖                         |
| 混合设备类型     | float attn_mask(CPU) + kpm(NPU) 报 RuntimeError   | 已覆盖                         |
| 入口关系         | 与 torch.nn.MultiheadAttention 为同一实现         | 已覆盖                         |

未覆盖项及原因：
- nested tensor query：merge_masks 仅在 attn_mask 非 None 时读取 query.shape，
  nested tensor 无 shape 展开语义，非本接口稳定契约，未覆盖。
- mask_type=0：实现中 attn_mask 存在即返回 mask_type=2（docstring 的 type 0 描述
  与实现不一致），以实际实现语义为准，不构造不存在的返回值。

注意：输出为 bool/float 掩码张量，CPU 侧为 Golden，NPU 输出移至 CPU 后用
     torch.testing.assert_close / torch.equal 对照；整数型 mask_type 用精确断言。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Resolve device only when tests actually run; keep default CI collection cheap.
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
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


import torch.nn.modules.activation as activation_mod  # noqa: E402


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestActivationMultiheadAttentionMergeMasks(TestCase):
    """Test torch.nn.modules.activation.MultiheadAttention.merge_masks (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)
        # merge_masks only reads num_heads and query shape; keep module on CPU.
        self.mha = activation_mod.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0,
                                                     batch_first=False)

    def _make_causal_attn(self, seq_len, batch_heads=None, device=None, dtype=torch.bool):
        # Lower-triangular bool mask, or float mask with -inf above the diagonal.
        if dtype == torch.bool:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            if batch_heads is None:
                return mask.to(device)
            return mask.unsqueeze(0).expand(batch_heads, seq_len, seq_len).contiguous().to(device)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
        mask.masked_fill_(~torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)),
                          float("-inf"))
        if batch_heads is None:
            return mask.to(device)
        return mask.unsqueeze(0).expand(batch_heads, seq_len, seq_len).contiguous().to(device)

    def test_merge_masks_none_none_returns_none(self):
        # Neither mask given: contract is (None, None).
        query = torch.randn(4, 5, 8)
        merged, mask_type = self.mha.merge_masks(None, None, query)
        self.assertIsNone(merged)
        self.assertIsNone(mask_type)

    def test_merge_masks_key_padding_only_returns_same_object(self):
        # key_padding_mask only: returned object must be the input itself, type 1.
        query = torch.randn(4, 5, 8)
        kpm = torch.tensor([[True, True, False, True, True]] * 4)
        merged, mask_type = self.mha.merge_masks(None, kpm, query)
        self.assertIs(merged, kpm)
        self.assertEqual(mask_type, 1)

    def test_merge_masks_attn_2d_expands_to_batch_heads(self):
        # 2D attn_mask: expanded to (batch, num_heads, seq, seq), type 2.
        query = torch.randn(4, 5, 8, device=self.device_name)
        attn = self._make_causal_attn(5, device=self.device_name)
        merged, mask_type = self.mha.merge_masks(attn, None, query)
        self.assertEqual(mask_type, 2)
        self.assertEqual(tuple(merged.shape), (4, 2, 5, 5))
        self.assertTrue(bool(torch.equal(merged.cpu(), attn.cpu().expand(4, 2, 5, 5))))

    def test_merge_masks_attn_3d_viewed_to_batch_heads(self):
        # 3D attn_mask with batch*num_heads rows: viewed as (batch, heads, seq, seq).
        query = torch.randn(4, 5, 8, device=self.device_name)
        attn = self._make_causal_attn(5, batch_heads=8, device=self.device_name)
        merged, mask_type = self.mha.merge_masks(attn, None, query)
        self.assertEqual(mask_type, 2)
        self.assertEqual(tuple(merged.shape), (4, 2, 5, 5))
        self.assertTrue(bool(torch.equal(merged.cpu(), attn.cpu().view(4, 2, 5, 5))))

    def test_merge_masks_bool_masks_combined_with_logical_or(self):
        # bool attn + bool kpm: merged = attn + kpm (bool add == logical or).
        query = torch.randn(4, 5, 8, device=self.device_name)
        attn = self._make_causal_attn(5, device=self.device_name)
        kpm = torch.tensor([[True, True, True, False, False]] * 4, device=self.device_name)
        merged, mask_type = self.mha.merge_masks(attn, kpm, query)
        self.assertEqual(mask_type, 2)
        self.assertEqual(merged.dtype, torch.bool)
        expected = attn.cpu().expand(4, 2, 5, 5) | kpm.cpu().view(4, 1, 1, 5).expand(4, 2, 5, 5)
        self.assertTrue(bool(torch.equal(merged.cpu(), expected)))
        # A kpm-only masked column must flip to True even where attn was False.
        self.assertTrue(bool(merged.cpu()[0, 0, 0, 2]))

    def test_merge_masks_float_attn_with_bool_kpm_adds_values(self):
        # float attn + bool kpm: merged is float; kpm True contributes 1.0, -inf preserved.
        query = torch.randn(4, 5, 8, device=self.device_name)
        attn = self._make_causal_attn(5, device=self.device_name, dtype=torch.float32)
        kpm = torch.tensor([[True, True, True, False, False]] * 4, device=self.device_name)
        merged, mask_type = self.mha.merge_masks(attn, kpm, query)
        self.assertEqual(mask_type, 2)
        self.assertEqual(merged.dtype, torch.float32)
        cpu_merged = merged.cpu()
        # (0, 0): attn 0.0, kpm True -> 1.0; kpm effect must be visible.
        self.assertEqual(cpu_merged[0, 0, 0, 0].item(), 1.0)
        # (0, 1): attn -inf, kpm True -> stays -inf (torch_npu assertEqual
        # rejects non-finite scalars, so compare the equality directly).
        self.assertTrue(cpu_merged[0, 0, 0, 1].item() == float("-inf"))
        # (0, 4): attn -inf, kpm False -> -inf.
        self.assertTrue(cpu_merged[0, 0, 0, 4].item() == float("-inf"))
        # (3, 3): attn 0.0 (tril row 3), kpm False -> 0.0.
        self.assertEqual(cpu_merged[0, 0, 3, 3].item(), 0.0)

    def test_merge_masks_npu_matches_cpu_golden(self):
        # Full parity check across both float/bool merge branches on device vs CPU.
        query_n = torch.randn(4, 5, 8, device=self.device_name)
        query_c = query_n.cpu()
        attn = self._make_causal_attn(5, device=self.device_name)
        kpm = torch.tensor([[True, True, True, False, False]] * 4, device=self.device_name)
        merged_n, type_n = self.mha.merge_masks(attn, kpm, query_n)
        merged_c, type_c = self.mha.merge_masks(attn.cpu(), kpm.cpu(), query_c)
        self.assertEqual(type_n, type_c)
        self.assertEqual(type_n, 2)
        torch.testing.assert_close(merged_n.cpu(), merged_c.cpu())
        attn_f = self._make_causal_attn(5, device=self.device_name, dtype=torch.float32)
        merged_fn, type_fn = self.mha.merge_masks(attn_f, kpm, query_n)
        merged_fc, _ = self.mha.merge_masks(attn_f.cpu(), kpm.cpu(), query_c)
        self.assertEqual(type_fn, 2)
        torch.testing.assert_close(merged_fn.cpu(), merged_fc.cpu())

    def test_merge_masks_seq_len_mismatch_raises(self):
        # attn_mask seq length differing from query seq length cannot be viewed.
        query = torch.randn(4, 5, 8, device=self.device_name)
        bad_attn = self._make_causal_attn(6, device=self.device_name)
        with self.assertRaises(RuntimeError):
            self.mha.merge_masks(bad_attn, None, query)

    def test_merge_masks_mixed_device_inputs_raise(self):
        # CPU float attn_mask + NPU kpm: torch add requires same device.
        query = torch.randn(4, 5, 8, device=self.device_name)
        attn_cpu = self._make_causal_attn(5, dtype=torch.float32)
        kpm = torch.tensor([[True, True, True, False, False]] * 4, device=self.device_name)
        with self.assertRaises(RuntimeError):
            self.mha.merge_masks(attn_cpu, kpm, query)

    def test_merge_masks_repeated_calls_deterministic(self):
        # Same inputs twice must produce identical merged mask (no hidden state).
        query = torch.randn(4, 5, 8, device=self.device_name)
        attn = self._make_causal_attn(5, device=self.device_name)
        kpm = torch.tensor([[True, False, False, True, True]] * 4, device=self.device_name)
        first, _ = self.mha.merge_masks(attn, kpm, query)
        second, _ = self.mha.merge_masks(attn, kpm, query)
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))

    def test_merge_masks_dual_entry_same_object(self):
        # torch.nn re-exports the same MultiheadAttention class from modules.activation.
        import torch.nn as nn
        self.assertIs(nn.MultiheadAttention, activation_mod.MultiheadAttention)
        self.assertIs(nn.MultiheadAttention.merge_masks,
                      activation_mod.MultiheadAttention.merge_masks)


if __name__ == "__main__":
    run_tests()
