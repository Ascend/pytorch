# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.dq_kv_order_spt 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.dq_kv_order_spt
API 签名：BlockMask.dq_kv_order_spt: bool | None（bool 型 KV 调度序标记）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | 未计算或显式张量序时为 None；bool 序时为 True/False | 已覆盖                         |
| 参数类型         | bool 或 None                                        | 已覆盖                         |
| 枚举选项         | dq_kv_order=True（SPT）/ False（升序）两分支        | 已覆盖                         |
| 传参与不传参     | create_block_mask 默认 True；from_kv_blocks 显式传  | 已覆盖                         |
| 正常传参场景     | 属性值与构建参数一致，CPU Golden 行为一致           | 已覆盖                         |
| 异常传参场景     | 不适用：bool 属性读取，非法类型在 dq_kv_order 路径  | 不适用                         |
| 混合设备类型     | 不适用：标量属性，无 Tensor 入参                    | 不适用                         |

未覆盖项及原因：
- 无

注意：dq_kv_order_spt 与 dq_kv_order 互斥；create_block_mask 在
     compute_dq_write_order=True 时按传入的 bool 调度序写入本属性。
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
class TestBlockMaskDqKvOrderSpt(TestCase):
    """Test BlockMask.dq_kv_order_spt flag contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dq_kv_order_spt_none_by_default(self):
        # Without dq write-order computation the SPT flag stays None.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32))
        self.assertIsNone(bm.dq_kv_order_spt)

    def test_dq_kv_order_spt_true_for_default_spt_order(self):
        # create_block_mask defaults dq_kv_order=True, stored as SPT flag True.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32), compute_dq_write_order=True)
        self.assertIs(bm.dq_kv_order_spt, True)
        self.assertIsNone(bm.dq_kv_order)

    def test_dq_kv_order_spt_false_for_ascending_order(self):
        # Explicit dq_kv_order=False must be recorded as False.
        bm = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                               BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                               dq_kv_order=False)
        self.assertIs(bm.dq_kv_order_spt, False)
        self.assertIsNone(bm.dq_kv_order)

    def test_dq_kv_order_spt_from_kv_blocks_bool_order(self):
        # from_kv_blocks stores a bool order in the SPT flag slot.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        bm_true = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                           seq_lengths=(4, 4), dq_kv_order=True)
        self.assertIs(bm_true.dq_kv_order_spt, True)
        self.assertIsNone(bm_true.dq_kv_order)
        bm_false = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                            seq_lengths=(4, 4), dq_kv_order=False)
        self.assertIs(bm_false.dq_kv_order_spt, False)

    def test_dq_kv_order_spt_none_for_tensor_order(self):
        # An explicit tensor scheduler order leaves the bool slot as None.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        order = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4), dq_kv_order=order)
        self.assertIsNone(bm.dq_kv_order_spt)
        self.assertIs(bm.dq_kv_order, order)

    def test_dq_kv_order_spt_matches_cpu_behavior(self):
        # CPU-built masks record the same flag values for both order branches.
        for order_value in (True, False):
            bm_dev = create_block_mask(causal_mask, 1, 1, 128, 128,
                                       device=self.device_name, BLOCK_SIZE=(32, 32),
                                       compute_dq_write_order=True, dq_kv_order=order_value)
            bm_cpu = create_block_mask(causal_mask, 1, 1, 128, 128, device="cpu",
                                       BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                       dq_kv_order=order_value)
            self.assertIs(bm_dev.dq_kv_order_spt, bm_cpu.dq_kv_order_spt)
            self.assertIs(bm_dev.dq_kv_order_spt, order_value)


if __name__ == "__main__":
    run_tests()
