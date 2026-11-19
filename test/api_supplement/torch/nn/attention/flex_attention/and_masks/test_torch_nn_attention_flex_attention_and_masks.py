# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.and_masks 接口功能正确性
API 名称：torch.nn.attention.flex_attention.and_masks
API 签名：and_masks(*mask_mods: Callable) -> Callable（交集组合掩码）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | 空 *args 返回恒真掩码                               | 已覆盖                         |
| 参数类型         | 可调用 mask_mod 变参；非可调用报 RuntimeError       | 已覆盖                         |
| 传参与不传参     | 单掩码/双掩码/三掩码组合                            | 已覆盖                         |
| 等价类/边界值    | 组合结果 == 逐个掩码按位与                          | 已覆盖                         |
| 正常传参场景     | 组合掩码经 create_mask 展开为稠密张量验证           | 已覆盖                         |
| 异常传参场景     | 混入字符串报 RuntimeError 并带原始提示              | 已覆盖                         |
| 混合设备类型     | 设备侧组合结果与 CPU Golden 一致                    | 已覆盖                         |
| 确定性           | 同输入重复执行组合掩码结果一致                      | 已覆盖                         |

未覆盖项及原因：
- 无

注意：组合结果为 bool 张量，CPU 侧为 Golden，设备侧输出移至 CPU 后用
     torch.equal 对照；mask_mod 直接调用路径用张量级按位与交叉验证。
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


from torch.nn.attention.flex_attention import and_masks  # noqa: E402
from torch.nn.attention.flex_attention import create_mask  # noqa: E402


def causal(b, h, q_idx, kv_idx):
    # Keep only positions at or below the diagonal.
    return q_idx >= kv_idx


def sliding(b, h, q_idx, kv_idx):
    # Keep a sliding window of four positions.
    return (q_idx - kv_idx) < 4


def even_columns(b, h, q_idx, kv_idx):
    # Keep even kv columns only.
    return kv_idx % 2 == 0


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAndMasks(TestCase):
    """Test flex_attention.and_masks combination contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_and_masks_returns_callable(self):
        # The combination of any number of mask_mods stays callable.
        self.assertTrue(callable(and_masks(causal)))
        self.assertTrue(callable(and_masks(causal, sliding)))

    def test_and_masks_single_mask_identity(self):
        # A single-mask combination must equal that mask alone.
        combined = and_masks(causal)
        dense_dev = create_mask(combined, 1, 1, 8, 8, device=self.device_name)
        dense_cpu = create_mask(causal, 1, 1, 8, 8, device="cpu")
        self.assertTrue(bool(torch.equal(dense_dev.cpu(), dense_cpu)))

    def test_and_masks_equals_elementwise_and(self):
        # Two-mask combination must equal the elementwise AND of both masks.
        combined = and_masks(causal, sliding)
        dense_dev = create_mask(combined, 1, 1, 8, 8, device=self.device_name)
        causal_dense = create_mask(causal, 1, 1, 8, 8, device="cpu")
        sliding_dense = create_mask(sliding, 1, 1, 8, 8, device="cpu")
        expected = causal_dense & sliding_dense
        self.assertEqual(dense_dev.dtype, torch.bool)
        self.assertTrue(bool(torch.equal(dense_dev.cpu(), expected)))
        # A falsifiable spot check: row 3 keeps causal but slides to 4-wide window.
        self.assertEqual(dense_dev.cpu()[0, 0, 3].tolist(),
                         [True, True, True, True, False, False, False, False])

    def test_and_masks_three_masks(self):
        # Three-mask combination equals the AND of all three dense masks.
        combined = and_masks(causal, sliding, even_columns)
        dense_dev = create_mask(combined, 1, 1, 8, 8, device=self.device_name)
        expected = (create_mask(causal, 1, 1, 8, 8, device="cpu")
                    & create_mask(sliding, 1, 1, 8, 8, device="cpu")
                    & create_mask(even_columns, 1, 1, 8, 8, device="cpu"))
        self.assertTrue(bool(torch.equal(dense_dev.cpu(), expected)))

    def test_and_masks_empty_args_always_true(self):
        # Combining zero masks yields a mask_mod that accepts everything.
        combined = and_masks()
        dense_dev = create_mask(combined, 2, 2, 4, 4, device=self.device_name)
        self.assertTrue(bool(dense_dev.cpu().all()))
        self.assertEqual(dense_dev.dtype, torch.bool)

    def test_and_masks_direct_call_on_device_tensors(self):
        # Calling the combined mask_mod directly must AND the callee results;
        # the CPU evaluation of the same combination serves as the golden.
        combined = and_masks(causal, sliding)
        q_idx = torch.tensor([2, 3], device=self.device_name)
        kv_idx = torch.tensor([1, 1], device=self.device_name)
        zero_dev = torch.zeros((), dtype=torch.long, device=self.device_name)
        zero_cpu = torch.zeros((), dtype=torch.long)
        result = combined(zero_dev, zero_dev, q_idx, kv_idx)
        golden = combined(zero_cpu, zero_cpu, q_idx.cpu(), kv_idx.cpu())
        self.assertEqual(result.dtype, torch.bool)
        self.assertTrue(bool(torch.equal(result.cpu(), golden)))
        # (q=2, kv=1): causal True, sliding True -> True; (q=3, kv=1): both True.
        self.assertTrue(bool(result.cpu()[0]))
        self.assertTrue(bool(result.cpu()[1]))

    def test_and_masks_rejects_non_callable(self):
        # Non-callable arguments must fail with the documented RuntimeError.
        with self.assertRaises(RuntimeError) as ctx:
            and_masks(causal, "not-a-mask")
        self.assertIn("All inputs should be callable", str(ctx.exception))

    def test_and_masks_repeated_evaluation_deterministic(self):
        # Repeated create_mask runs over the same combination are identical.
        combined = and_masks(causal, sliding)
        first = create_mask(combined, 1, 1, 8, 8, device=self.device_name)
        second = create_mask(combined, 1, 1, 8, 8, device=self.device_name)
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
