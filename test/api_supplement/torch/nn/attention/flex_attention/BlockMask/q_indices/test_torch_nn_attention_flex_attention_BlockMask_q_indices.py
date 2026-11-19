# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.q_indices 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.q_indices
API 签名：BlockMask.q_indices: Tensor | None（反向传播用的 Q 维块索引）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | compute_q_blocks=True 生成 / False 置 None          | 已覆盖                         |
| 参数类型         | Tensor（int32）或 None                              | 已覆盖                         |
| 等价类/边界值    | create_block_mask 与 from_kv_blocks 两种来源        | 已覆盖                         |
| 正常传参场景     | 设备、形状、数值与 CPU Golden 一致                  | 已覆盖                         |
| 异常传参场景     | 不适用：实例属性读取，无入参异常路径                | 不适用                         |
| 混合设备类型     | 不适用：属性读取，无多 Tensor 入参                  | 不适用                         |
| 确定性           | 重复构建同掩码 q_indices 一致                       | 已覆盖                         |

未覆盖项及原因：
- 无

注意：q_indices 为反向 Q 维块索引张量，CPU 构建的同构 BlockMask 为 Golden，
     设备侧张量移至 CPU 后用 torch.equal 逐元素对照。
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
class TestBlockMaskQIndices(TestCase):
    """Test BlockMask.q_indices generation contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_q_indices_present_after_create_block_mask(self):
        # create_block_mask computes q side by default; it lives on the mask device.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertIsNotNone(bm.q_indices)
        self.assertIsInstance(bm.q_indices, torch.Tensor)
        self.assertEqual(bm.q_indices.device.type, self.device_name)
        self.assertEqual(bm.q_indices.dtype, torch.int32)

    def test_q_indices_matches_cpu_golden(self):
        # Device-built q_indices must equal the CPU-built golden element-wise.
        bm_dev = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32))
        bm_cpu = create_block_mask(causal_mask, 1, 1, 128, 128, device="cpu",
                                   BLOCK_SIZE=(32, 32))
        self.assertEqual(tuple(bm_dev.q_indices.shape), tuple(bm_cpu.q_indices.shape))
        self.assertEqual(tuple(bm_dev.q_indices.shape), (1, 1, 4, 4))
        self.assertTrue(bool(torch.equal(bm_dev.q_indices.cpu(), bm_cpu.q_indices)))

    def test_q_indices_hand_built_from_kv_blocks(self):
        # Hand-built kv lists transpose into q_indices on the kv-column axis.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None,
                                      BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertIsNotNone(bm.q_indices)
        self.assertEqual(tuple(bm.q_indices.shape), (1, 1, 2, 2))
        self.assertEqual(bm.q_indices.dtype, torch.int32)
        kv_num_c = kv_num.cpu()
        kv_idx_c = kv_idx.cpu()
        bm_golden = BlockMask.from_kv_blocks(kv_num_c, kv_idx_c, None, None,
                                             BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertTrue(bool(torch.equal(bm.q_indices.cpu(), bm_golden.q_indices)))
        # q_indices rows are the kv-column lists; kv lists must stay untouched.
        self.assertTrue(bool(torch.equal(bm.kv_num_blocks.cpu(), kv_num_c)))
        self.assertTrue(bool(torch.equal(bm.kv_indices.cpu(), kv_idx_c)))

    def test_q_indices_none_when_compute_q_blocks_disabled(self):
        # compute_q_blocks=False leaves the generated q side as None.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4), compute_q_blocks=False)
        self.assertIsNone(bm.q_indices)
        self.assertIsNone(bm.q_num_blocks)

    def test_q_indices_repeated_build_deterministic(self):
        # Building the same mask twice must yield identical q_indices.
        first = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                  BLOCK_SIZE=(32, 32)).q_indices
        second = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32)).q_indices
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
