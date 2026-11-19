# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.dq_kv_order 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.dq_kv_order
API 签名：BlockMask.dq_kv_order: Tensor | None（显式 KV 调度序张量）

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 空/非空          | bool 调度序或未计算时为 None；显式张量时非 None     | 已覆盖                         |
| 参数类型         | Tensor 或 None（bool 调度序存入 dq_kv_order_spt）   | 已覆盖                         |
| 枚举选项         | from_kv_blocks 接受 Tensor / bool / None            | 已覆盖                         |
| 正常传参场景     | 张量原样存储（同一性）、bool 路径置 None            | 已覆盖                         |
| 异常传参场景     | 非法类型（str）报 ValueError；create_block_mask     | 已覆盖                         |
|                  | 传非 bool 报 ValueError                             |                                |
| 混合设备类型     | 不适用：实例属性读取，无多 Tensor 入参              | 不适用                         |

未覆盖项及原因：
- 无

注意：dq_kv_order 张量与 dq_kv_order_spt 互斥存储；create_block_mask 仅接受
     bool 调度序，显式张量路径由 from_kv_blocks 提供。
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
class TestBlockMaskDqKvOrder(TestCase):
    """Test BlockMask.dq_kv_order storage contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _hand_built_inputs(self):
        # Consistent kv lists: two q-rows over two kv columns.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        return kv_num, kv_idx

    def test_dq_kv_order_none_for_bool_and_default_paths(self):
        # create_block_mask only accepts bool orders, so dq_kv_order stays None.
        bm_default = create_block_mask(causal_mask, 1, 1, 128, 128,
                                       device=self.device_name, BLOCK_SIZE=(32, 32))
        self.assertIsNone(bm_default.dq_kv_order)
        bm_spt = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                   BLOCK_SIZE=(32, 32), compute_dq_write_order=True,
                                   dq_kv_order=True)
        self.assertIsNone(bm_spt.dq_kv_order)
        self.assertTrue(bm_spt.dq_kv_order_spt)

    def test_dq_kv_order_tensor_stored_verbatim(self):
        # from_kv_blocks keeps an explicit scheduler-order tensor by reference.
        kv_num, kv_idx = self._hand_built_inputs()
        order = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4), dq_kv_order=order)
        self.assertIs(bm.dq_kv_order, order)
        self.assertEqual(bm.dq_kv_order.device.type, self.device_name)
        self.assertTrue(bool(torch.equal(bm.dq_kv_order.cpu(), order.cpu())))
        # The bool slot stays empty for the tensor path.
        self.assertIsNone(bm.dq_kv_order_spt)

    def test_dq_kv_order_none_for_bool_order(self):
        # A bool order is stored in dq_kv_order_spt, leaving dq_kv_order None.
        kv_num, kv_idx = self._hand_built_inputs()
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                      seq_lengths=(4, 4), dq_kv_order=False)
        self.assertIsNone(bm.dq_kv_order)
        self.assertFalse(bm.dq_kv_order_spt)
        bm_default = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None,
                                              BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        self.assertIsNone(bm_default.dq_kv_order)
        self.assertIsNone(bm_default.dq_kv_order_spt)

    def test_dq_kv_order_invalid_type_raises_value_error(self):
        # A non-bool non-tensor order must be rejected by from_kv_blocks.
        kv_num, kv_idx = self._hand_built_inputs()
        with self.assertRaises(ValueError):
            BlockMask.from_kv_blocks(kv_num, kv_idx, None, None, BLOCK_SIZE=(2, 2),
                                     seq_lengths=(4, 4), dq_kv_order="ascending")

    def test_dq_kv_order_create_block_mask_requires_bool(self):
        # create_block_mask rejects explicit tensor orders with ValueError.
        order = torch.tensor([[0, 1]], dtype=torch.int32, device=self.device_name)
        with self.assertRaises(ValueError):
            create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                              BLOCK_SIZE=(32, 32), dq_kv_order=order)


if __name__ == "__main__":
    run_tests()
