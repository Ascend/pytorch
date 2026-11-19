# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.dq_write_order_full 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.dq_write_order_full
API 签名：BlockMask.dq_write_order_full: Tensor | None（全块部分的确定性 dQ 写序元数据）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | separate_full_blocks 开/关与预计算三种来源          | 已覆盖                         |
| 参数类型         | Tensor（int32）或 None                              | 已覆盖                         |
| 传参与不传参     | compute_dq_write_order 显式开启                     | 已覆盖                         |
| 等价类/边界值    | 无全块时为 None 的边界                              | 已覆盖                         |
| 正常传参场景     | 设备、形状、数值与 CPU Golden 一致                  | 已覆盖                         |
| 异常传参场景     | 单独提供 dq_write_order_full 报 ValueError          | 已覆盖                         |
| 混合设备类型     | 不适用：实例属性读取，无多 Tensor 入参              | 不适用                         |
| 确定性           | 重复构建同掩码 dq_write_order_full 一致             | 已覆盖                         |

未覆盖项及原因：
- 无

注意：dq_write_order_full 依赖 separate_full_blocks=True 的全块列表；CPU 构建
     的同构 BlockMask 为 Golden，设备侧张量移至 CPU 后用 torch.equal 对照。
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
class TestBlockMaskDqWriteOrderFull(TestCase):
    """Test BlockMask.dq_write_order_full generation contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dq_write_order_full_none_by_default(self):
        # Without compute_dq_write_order the full-block write order stays None.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertIsNone(bm.dq_write_order_full)

    def test_dq_write_order_full_computed_with_full_blocks(self):
        # separate_full_blocks=True (default) yields full-block write-order metadata.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                               dq_kv_order=True)
        self.assertIsNotNone(bm.dq_write_order_full)
        self.assertIsInstance(bm.dq_write_order_full, torch.Tensor)
        self.assertEqual(bm.dq_write_order_full.device.type, self.device_name)
        self.assertEqual(bm.dq_write_order_full.dtype, torch.int32)
        self.assertEqual(tuple(bm.dq_write_order_full.shape), (1, 1, 4, 4))

    def test_dq_write_order_full_matches_cpu_golden(self):
        # Device-computed full write order must equal the CPU-computed golden.
        bm_dev = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True)
        bm_cpu = create_block_mask(causal_mask, 1, 1, 128, 128, device="cpu",
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True)
        self.assertTrue(bool(torch.equal(bm_dev.dq_write_order_full.cpu(),
                                         bm_cpu.dq_write_order_full)))

    def test_dq_write_order_full_none_without_separate_full_blocks(self):
        # Merging full blocks into partial ones leaves the full write order None.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                               separate_full_blocks=False)
        self.assertIsNone(bm.dq_write_order_full)
        # The partial-block write order is still computed in this mode.
        self.assertIsNotNone(bm.dq_write_order)

    def test_dq_write_order_full_precomputed_stored_verbatim(self):
        # from_kv_blocks stores caller-provided full write-order metadata as-is.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        provided = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.int32,
                                device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4),
                                      dq_write_order=torch.zeros(
                                          1, 1, 2, 2, dtype=torch.int32,
                                          device=self.device_name),
                                      dq_write_order_full=provided)
        self.assertIs(bm.dq_write_order_full, provided)
        self.assertTrue(bool(torch.equal(bm.dq_write_order_full.cpu(), provided.cpu())))

    def test_dq_write_order_full_alone_raises_value_error(self):
        # dq_write_order_full without dq_write_order must raise ValueError.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        with self.assertRaises(ValueError):
            BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                     seq_lengths=(4, 4),
                                     dq_write_order_full=torch.zeros(
                                         1, 1, 2, 2, dtype=torch.int32,
                                         device=self.device_name))

    def test_dq_write_order_full_repeated_compute_deterministic(self):
        # Repeated computation on the same mask must be bit-identical.
        first = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                  BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                  dq_kv_order=True).dq_write_order_full
        second = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True).dq_write_order_full
        self.assertTrue(bool(torch.equal(first.cpu(), second.cpu())))


if __name__ == "__main__":
    run_tests()
