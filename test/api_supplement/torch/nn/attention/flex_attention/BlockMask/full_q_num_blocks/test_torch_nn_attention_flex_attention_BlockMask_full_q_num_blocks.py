# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.full_q_num_blocks 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.full_q_num_blocks
API 签名：BlockMask.full_q_num_blocks: Tensor | None（反向传播用的全块 Q 维块计数）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | 提供full_kv时生成；未提供或关闭生成时为 None        | 已覆盖                         |
| 参数类型         | Tensor（int32）或 None                              | 已覆盖                         |
| 等价类/边界值    | create_block_mask 与 from_kv_blocks 两种来源        | 已覆盖                         |
| 正常传参场景     | 设备、形状、数值与 CPU Golden 一致                  | 已覆盖                         |
| 异常传参场景     | 不适用：实例属性读取，无入参异常路径                | 不适用                         |
| 混合设备类型     | 不适用：属性读取，无多 Tensor 入参                  | 不适用                         |
| 确定性           | 重复构建同掩码 full_q_num_blocks 一致               | 已覆盖                         |

未覆盖项及原因：
- 无

注意：full_q_num_blocks 由 full_kv_* 转置生成，CPU 构建的同构 BlockMask 为
     Golden，设备侧张量移至 CPU 后用 torch.equal 逐元素对照。
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
    # Standard causal pattern used for both device and CPU golden builds.
    return q_idx >= kv_idx


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBlockMaskFullQNumBlocks(TestCase):
    """Test BlockMask.full_q_num_blocks generation contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_full_q_num_blocks_present_after_create_block_mask(self):
        # separate_full_blocks defaults to True, so the generated full q side exists.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertIsNotNone(bm.full_q_num_blocks)
        self.assertIsInstance(bm.full_q_num_blocks, torch.Tensor)
        self.assertEqual(bm.full_q_num_blocks.device.type, self.device_name)
        self.assertEqual(bm.full_q_num_blocks.dtype, torch.int32)

    def test_full_q_num_blocks_matches_cpu_golden(self):
        # Device-built full_q_num_blocks must equal the CPU-built golden.
        bm_dev = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32))
        bm_cpu = create_block_mask(causal_mask, 1, 1, 128, 128, device="cpu",
                                   BLOCK_SIZE=(32, 32))
        self.assertEqual(tuple(bm_dev.full_q_num_blocks.shape),
                         tuple(bm_cpu.full_q_num_blocks.shape))
        # Rows are kv columns: counts per column, max three full blocks
        # (the diagonal block is partial on a 4x4 causal grid).
        self.assertEqual(tuple(bm_dev.full_q_num_blocks.shape), (1, 1, 4))
        # Indices keep the full kv-column width; entries beyond the count
        # in each row are undefined by the BlockMask contract.
        self.assertEqual(tuple(bm_dev.full_q_indices.shape), (1, 1, 4, 4))
        self.assertTrue(bool(torch.equal(bm_dev.full_q_num_blocks.cpu(),
                                         bm_cpu.full_q_num_blocks)))

    def test_full_q_num_blocks_none_without_full_kv(self):
        # from_kv_blocks without full_kv_* must leave the full q side as None.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None,
                                      BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertIsNone(bm.full_q_num_blocks)
        self.assertIsNone(bm.full_q_indices)

    def test_full_q_num_blocks_transposed_from_full_kv(self):
        # Hand-built full kv lists must transpose into full_q_num_blocks on device.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        full_num = torch.tensor([[[1, 1]]], device=self.device_name)
        full_idx = torch.tensor([[[[1, 0], [0, 0]]]], device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx,
                                      BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertIsNotNone(bm.full_q_num_blocks)
        # Rows are kv columns: two kv columns -> counts per column, not blocks.
        self.assertEqual(tuple(bm.full_q_num_blocks.shape), (1, 1, 2))
        self.assertEqual(tuple(bm.full_q_indices.shape), (1, 1, 2, 2))
        kv_num_c = kv_num.cpu()
        kv_idx_c = kv_idx.cpu()
        full_num_c = full_num.cpu()
        full_idx_c = full_idx.cpu()
        bm_golden = BlockMask.from_kv_blocks(kv_num_c, kv_idx_c, full_num_c, full_idx_c,
                                             BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertTrue(bool(torch.equal(bm.full_q_num_blocks.cpu(),
                                         bm_golden.full_q_num_blocks)))

    def test_full_q_num_blocks_none_when_compute_q_blocks_disabled(self):
        # compute_q_blocks=False skips the generated q side entirely.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        full_num = torch.tensor([[[1, 1]]], device=self.device_name)
        full_idx = torch.tensor([[[[1, 0], [0, 0]]]], device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4), compute_q_blocks=False)
        self.assertIsNone(bm.full_q_num_blocks)
        self.assertIsNone(bm.full_q_indices)

    def test_full_q_num_blocks_repeated_build_deterministic(self):
        # Building the same mask twice must yield identical full_q_num_blocks.
        first = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                  BLOCK_SIZE=(32, 32)).full_q_num_blocks
        second = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32)).full_q_num_blocks
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
