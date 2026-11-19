# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.dq_write_order 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.dq_write_order
API 签名：BlockMask.dq_write_order: Tensor | None（确定性 dQ 写序元数据）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | compute_dq_write_order 开/关与预计算三种来源        | 已覆盖                         |
| 参数类型         | Tensor（int32）或 None                              | 已覆盖                         |
| 枚举选项         | dq_kv_order=True/False 两种调度序                   | 已覆盖                         |
| 正常传参场景     | 设备、形状、数值与 CPU Golden 一致                  | 已覆盖                         |
| 异常传参场景     | dq_write_order_full 缺 dq_write_order 报 ValueError | 已覆盖                         |
| 混合设备类型     | 不适用：实例属性读取，无多 Tensor 入参              | 不适用                         |
| 确定性           | 重复构建同掩码 dq_write_order 一致                  | 已覆盖                         |

未覆盖项及原因：
- 无

注意：dq_write_order 由 create_block_mask(compute_dq_write_order=True) 计算，
     CPU 构建的同构 BlockMask 为 Golden，设备侧张量移至 CPU 后用 torch.equal
     逐元素对照；True/False 调度序产生不同写序，作为可证伪断言。
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
class TestBlockMaskDqWriteOrder(TestCase):
    """Test BlockMask.dq_write_order generation contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dq_write_order_none_by_default(self):
        # Without compute_dq_write_order the write-order metadata stays None.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertIsNone(bm.dq_write_order)

    def test_dq_write_order_computed_with_spt_order(self):
        # compute_dq_write_order=True with dq_kv_order=True (SPT) fills metadata.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                               dq_kv_order=True)
        self.assertIsNotNone(bm.dq_write_order)
        self.assertIsInstance(bm.dq_write_order, torch.Tensor)
        self.assertEqual(bm.dq_write_order.device.type, self.device_name)
        self.assertEqual(bm.dq_write_order.dtype, torch.int32)
        self.assertEqual(tuple(bm.dq_write_order.shape), (1, 1, 4, 4))

    def test_dq_write_order_matches_cpu_golden(self):
        # Device-computed write order must equal the CPU-computed golden.
        bm_dev = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True)
        bm_cpu = create_block_mask(causal_mask, 1, 1, 128, 128, device="cpu",
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True)
        self.assertTrue(bool(torch.equal(bm_dev.dq_write_order.cpu(),
                                         bm_cpu.dq_write_order)))

    def test_dq_write_order_scheduler_orders_differ(self):
        # Ascending (False) and SPT (True) orders must produce different tensors.
        bm_asc = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=False)
        bm_spt = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True)
        self.assertIsNotNone(bm_asc.dq_write_order)
        self.assertFalse(bool(torch.equal(bm_asc.dq_write_order.cpu(),
                                          bm_spt.dq_write_order.cpu())))

    def test_dq_write_order_precomputed_stored_verbatim(self):
        # from_kv_blocks stores a caller-provided dq_write_order tensor as-is.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        provided = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.int32,
                                device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4), dq_write_order=provided)
        self.assertIs(bm.dq_write_order, provided)
        self.assertTrue(bool(torch.equal(bm.dq_write_order.cpu(), provided.cpu())))

    def test_dq_write_order_full_requires_dq_write_order(self):
        # Providing dq_write_order_full alone must raise ValueError.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        with self.assertRaises(ValueError):
            BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                     seq_lengths=(4, 4),
                                     dq_write_order_full=torch.zeros(
                                         1, 1, 2, 2, dtype=torch.int32,
                                         device=self.device_name))

    def test_dq_write_order_repeated_compute_deterministic(self):
        # Repeated computation on the same mask must be bit-identical.
        first = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                  BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                  dq_kv_order=True).dq_write_order
        second = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True).dq_write_order
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
