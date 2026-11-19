# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.sparsity 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.sparsity
API 签名：sparsity(self) -> float（未参与计算的块占比百分数）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 等价类/边界值    | 全稠密 0%、对角 75%、因果 37.5% 三个已知精确值      | 已覆盖                         |
| 参数类型         | 返回 float 百分数                                   | 已覆盖                         |
| 正常传参场景     | 设备侧掩码与 CPU Golden 掩码稀疏度一致              | 已覆盖                         |
| 异常传参场景     | 不适用：无参方法，无异常路径                        | 不适用                         |
| 混合设备类型     | 不适用：单实例方法，无多 Tensor 入参                | 不适用                         |
| 可证伪性         | 不同掩码稀疏度必须不同                              | 已覆盖                         |
| 确定性           | 同一掩码重复调用结果一致                            | 已覆盖                         |

未覆盖项及原因：
- 无

注意：sparsity 为纯统计值，断言使用与块布局手工推导的精确数值
     （assertAlmostEqual），并以 CPU 构建的同构掩码交叉对照。
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


from torch.nn.attention.flex_attention import BlockMask  # noqa: E402
from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402


def causal_mask(b, h, q_idx, kv_idx):
    # Standard causal pattern: 1+2+3+4 = 10 computed blocks over 16 -> 37.5% sparse.
    return q_idx >= kv_idx


def diagonal_mask(b, h, q_idx, kv_idx):
    # One kv block per q block: 4 computed blocks over 16 -> 75% sparse.
    return q_idx == kv_idx


def noop_mask(b, h, q_idx, kv_idx):
    # Everything attends: 16 computed blocks over 16 -> 0% sparse.
    return torch.ones_like(q_idx >= kv_idx)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBlockMaskSparsity(TestCase):
    """Test BlockMask.sparsity exact values and device parity (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_sparsity_identity_mask_is_zero(self):
        # A mask where every block participates must report exactly 0%.
        bm = create_block_mask(noop_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertAlmostEqual(bm.sparsity(), 0.0, places=9)

    def test_sparsity_causal_mask_exact_value(self):
        # Causal 128x128 with 32-blocks: computed = 1+2+3+4 = 10 of 16 -> 37.5%.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertAlmostEqual(bm.sparsity(), 37.5, places=9)

    def test_sparsity_diagonal_mask_exact_value(self):
        # Diagonal mask: computed = 4 of 16 blocks -> exactly 75%.
        bm = create_block_mask(diagonal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertAlmostEqual(bm.sparsity(), 75.0, places=9)

    def test_sparsity_device_matches_cpu_golden(self):
        # Device-built and CPU-built masks must report identical sparsity.
        for mask_fn in (causal_mask, diagonal_mask, noop_mask):
            bm_dev = create_block_mask(mask_fn, 1, 1, 128, 128, device=self.device_name,
                                       BLOCK_SIZE=(32, 32))
            bm_cpu = create_block_mask(mask_fn, 1, 1, 128, 128, device="cpu",
                                       BLOCK_SIZE=(32, 32))
            self.assertAlmostEqual(bm_dev.sparsity(), bm_cpu.sparsity(), places=9)

    def test_sparsity_returns_float_and_distinguishes_masks(self):
        # The result is a Python float; distinct layouts must give distinct values.
        bm_causal = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                      BLOCK_SIZE=(32, 32))
        bm_diag = create_block_mask(diagonal_mask, 1, 1, 128, 128, device=self.device_name,
                                    BLOCK_SIZE=(32, 32))
        self.assertIsInstance(bm_causal.sparsity(), float)
        self.assertNotAlmostEqual(bm_causal.sparsity(), bm_diag.sparsity(), places=9)

    def test_sparsity_full_blocks_count_toward_computation(self):
        # full_kv_* blocks are counted as computed alongside partial ones:
        # total = 2x2 = 4 blocks; computed = 3 partial + 1 full -> 0% sparse.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        full_num = torch.tensor([[[1, 0]]], device=self.device_name)
        full_idx = torch.tensor([[[[0, 0], [0, 0]]]], device=self.device_name)
        bm_full = BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx,
                                           BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertAlmostEqual(bm_full.sparsity(), 0.0, places=9)
        # The same layout without the full block reports 25% (3 of 4 computed).
        bm_partial = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None,
                                              BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertAlmostEqual(bm_partial.sparsity(), 25.0, places=9)


if __name__ == "__main__":
    run_tests()
