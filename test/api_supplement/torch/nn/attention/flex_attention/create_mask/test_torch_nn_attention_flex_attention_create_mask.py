# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.create_mask 接口功能正确性
API 名称：torch.nn.attention.flex_attention.create_mask
API 签名：create_mask(mod_fn, B, H, Q_LEN, KV_LEN, device=None) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 参数类型         | mask_mod 与 score_mod 两类 mod_fn                   | 已覆盖                         |
| 传参与不传参     | B/H 为 None 默认 1；device 为 None 用当前加速器     | 已覆盖                         |
| 等价类/边界值    | 与手工 tril 参考掩码逐元素对照                      | 已覆盖                         |
| 正常传参场景     | 形状 (B, H, Q, KV)、dtype bool、设备契约            | 已覆盖                         |
| 异常传参场景     | 不适用：非法 mod_fn 在 _get_mod_type 断言属内部契约 | 不适用                         |
| 混合设备类型     | 不适用：单 device 参数构造，无多 Tensor 入参        | 不适用                         |
| 确定性           | 同参数重复调用结果一致                              | 已覆盖                         |

未覆盖项及原因：
- 非 mask_mod/score_mod 的 mod_fn：内部 _get_mod_type 断言失败属内部实现
  细节，非稳定公开契约，未覆盖。

注意：输出为 bool 掩码张量，CPU 侧为 Golden，设备侧输出移至 CPU 后用
     torch.equal 逐元素对照，并与手工构造的 tril 参考值交叉验证。
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


from torch.nn.attention.flex_attention import create_mask  # noqa: E402


def causal_mask_mod(b, h, q_idx, kv_idx):
    # Keep only positions at or below the diagonal.
    return q_idx >= kv_idx


def causal_score_mod(score, b, h, q_idx, kv_idx):
    # Mask via score modification: out-of-triangle positions become -inf.
    return torch.where(q_idx >= kv_idx, score, float("-inf"))


def identity_score_mod(score, b, h, q_idx, kv_idx):
    # Never masks anything.
    return score


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCreateMask(TestCase):
    """Test flex_attention.create_mask dense-mask contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _expected_causal(self, q_len, kv_len):
        # Manual reference: mask[b, h, q, kv] = q >= kv.
        q_pos = torch.arange(q_len).unsqueeze(-1)
        kv_pos = torch.arange(kv_len).unsqueeze(0)
        return (q_pos >= kv_pos).unsqueeze(0).unsqueeze(0)

    def test_create_mask_mask_mod_matches_manual_reference(self):
        # The mask_mod path must reproduce a hand-built lower-triangular mask.
        dense_dev = create_mask(causal_mask_mod, 1, 1, 8, 8, device=self.device_name)
        expected = self._expected_causal(8, 8)
        self.assertEqual(dense_dev.dtype, torch.bool)
        self.assertEqual(tuple(dense_dev.shape), (1, 1, 8, 8))
        self.assertTrue(bool(torch.equal(dense_dev.cpu(), expected)))
        # Falsifiable spot check: row 0 only attends to kv 0.
        self.assertEqual(dense_dev.cpu()[0, 0, 0].tolist(),
                         [True] + [False] * 7)

    def test_create_mask_mask_mod_matches_cpu_golden(self):
        # Device-built and CPU-built masks are element-wise equal.
        dense_dev = create_mask(causal_mask_mod, 2, 3, 16, 24, device=self.device_name)
        dense_cpu = create_mask(causal_mask_mod, 2, 3, 16, 24, device="cpu")
        self.assertEqual(dense_dev.device.type, self.device_name)
        self.assertEqual(tuple(dense_dev.shape), (2, 3, 16, 24))
        self.assertTrue(bool(torch.equal(dense_dev.cpu(), dense_cpu)))

    def test_create_mask_score_mod_neg_inf_becomes_false(self):
        # The score_mod path turns -inf scores into masked (False) positions.
        dense_dev = create_mask(causal_score_mod, 1, 1, 4, 4, device=self.device_name)
        expected = self._expected_causal(4, 4)
        self.assertTrue(bool(torch.equal(dense_dev.cpu(), expected)))
        identity = create_mask(identity_score_mod, 1, 1, 4, 4, device=self.device_name)
        self.assertTrue(bool(identity.cpu().all()))

    def test_create_mask_default_b_h_are_one(self):
        # B=None and H=None collapse to a (1, 1, Q, KV) mask.
        dense_dev = create_mask(causal_mask_mod, None, None, 6, 6, device=self.device_name)
        self.assertEqual(tuple(dense_dev.shape), (1, 1, 6, 6))

    def test_create_mask_default_device_follows_accelerator(self):
        # device=None selects the current accelerator (npu/cuda) or falls back.
        dense_dev = create_mask(causal_mask_mod, 1, 1, 4, 4)
        self.assertEqual(dense_dev.device.type, self.device_name)

    def test_create_mask_repeated_calls_deterministic(self):
        # Two runs with identical inputs must yield identical masks.
        first = create_mask(causal_mask_mod, 2, 2, 12, 12, device=self.device_name)
        second = create_mask(causal_mask_mod, 2, 2, 12, 12, device=self.device_name)
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
