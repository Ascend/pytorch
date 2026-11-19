# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.as_tuple 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.as_tuple
API 签名：as_tuple(self, flatten: bool = True) -> tuple[Any, ...]

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 枚举选项         | flatten=True / flatten=False 两种返回结构           | 已覆盖                         |
| 参数类型         | 返回 tuple，元素含 int/Tensor/bool/function/tuple   | 已覆盖                         |
| 传参与不传参     | 默认 flatten=True；显式 False                       | 已覆盖                         |
| 等价类/边界值    | 有/无 full_kv、有/无 dq 元数据的掩码                | 已覆盖                         |
| 正常传参场景     | 元素顺序、对象同一性、数值与属性一致                | 已覆盖                         |
| 异常传参场景     | 不适用：无稳定异常路径（bool 之外入参无契约）       | 不适用                         |
| 混合设备类型     | 不适用：单实例方法，无多 Tensor 入参                | 不适用                         |

未覆盖项及原因：
- 无

注意：as_tuple 中的 Tensor 元素与 CPU Golden 构建的掩码逐元素
     torch.equal 对照；元素与实例属性使用 assertIs 同一性断言。
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
class TestBlockMaskAsTuple(TestCase):
    """Test BlockMask.as_tuple structure and ordering (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _build_mask(self, device_name):
        # Rich mask: separate full blocks plus dq write-order metadata.
        return create_block_mask(
            causal_mask, 1, 1, 128, 128, device=device_name,
            BLOCK_SIZE=(32, 32), compute_dq_write_order=True, dq_kv_order=True)

    def test_as_tuple_flatten_length_and_order(self):
        # flatten=True returns 17 elements in documented attribute order.
        bm = self._build_mask(self.device_name)
        parts = bm.as_tuple()
        self.assertEqual(len(parts), 17)
        self.assertEqual(parts[0], 128)
        self.assertEqual(parts[1], 128)
        self.assertIs(parts[2], bm.kv_num_blocks)
        self.assertIs(parts[3], bm.kv_indices)
        self.assertIs(parts[4], bm.full_kv_num_blocks)
        self.assertIs(parts[5], bm.full_kv_indices)
        self.assertIs(parts[6], bm.q_num_blocks)
        self.assertIs(parts[7], bm.q_indices)
        self.assertIs(parts[8], bm.full_q_num_blocks)
        self.assertIs(parts[9], bm.full_q_indices)
        self.assertIs(parts[10], bm.dq_write_order)
        self.assertIs(parts[11], bm.dq_write_order_full)
        self.assertIs(parts[12], bm.dq_kv_order)
        self.assertEqual(parts[13], bm.dq_kv_order_spt)
        self.assertEqual(parts[14], bm.BLOCK_SIZE[0])
        self.assertEqual(parts[15], bm.BLOCK_SIZE[1])
        self.assertIs(parts[16], bm.mask_mod)

    def test_as_tuple_flatten_types(self):
        # Scalar slots are int/bool; dq_kv_order slot is None for SPT-ordered masks.
        bm = self._build_mask(self.device_name)
        parts = bm.as_tuple()
        self.assertIsInstance(parts[0], int)
        self.assertIsInstance(parts[1], int)
        self.assertIsInstance(parts[2], torch.Tensor)
        self.assertIsInstance(parts[13], bool)
        self.assertIsNone(parts[12])
        self.assertIsInstance(parts[14], int)
        self.assertIsInstance(parts[16], type(causal_mask))

    def test_as_tuple_non_flatten_groups_block_sizes(self):
        # flatten=False returns 15 elements: seq_lengths/BLOCK_SIZE stay grouped.
        bm = self._build_mask(self.device_name)
        parts = bm.as_tuple(flatten=False)
        self.assertEqual(len(parts), 15)
        self.assertEqual(parts[0], (128, 128))
        self.assertIs(parts[1], bm.kv_num_blocks)
        self.assertEqual(parts[13], bm.BLOCK_SIZE)
        self.assertIs(parts[14], bm.mask_mod)

    def test_as_tuple_tensor_elements_match_attributes(self):
        # Every tensor slot must carry identical values to the instance attribute.
        bm = self._build_mask(self.device_name)
        parts = bm.as_tuple()
        for slot, attr in ((2, "kv_num_blocks"), (3, "kv_indices"),
                           (6, "q_num_blocks"), (7, "q_indices"),
                           (8, "full_q_num_blocks"), (9, "full_q_indices"),
                           (10, "dq_write_order"), (11, "dq_write_order_full")):
            expected = getattr(bm, attr)
            self.assertTrue(bool(torch.equal(parts[slot].cpu(), expected.cpu())), attr)

    def test_as_tuple_matches_cpu_golden(self):
        # Device-built and CPU-built masks produce element-wise equal tuples.
        bm_dev = self._build_mask(self.device_name)
        bm_cpu = self._build_mask("cpu")
        parts_dev = bm_dev.as_tuple()
        parts_cpu = bm_cpu.as_tuple()
        self.assertEqual(parts_dev[:2], parts_cpu[:2])
        for idx in range(2, 12):
            self.assertTrue(bool(torch.equal(parts_dev[idx].cpu(), parts_cpu[idx].cpu())), idx)
        self.assertEqual(parts_dev[12], parts_cpu[12])
        self.assertEqual(parts_dev[13], parts_cpu[13])
        self.assertEqual(parts_dev[14:16], parts_cpu[14:16])

    def test_as_tuple_hand_built_mask_roundtrip(self):
        # A from_kv_blocks mask with full blocks exposes full_q slots in the tuple.
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        full_num = torch.tensor([[[1, 1]]], device=self.device_name)
        full_idx = torch.tensor([[[[1, 0], [0, 0]]]], device=self.device_name)
        bm = BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx,
                                      BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        parts = bm.as_tuple()
        self.assertEqual(len(parts), 17)
        self.assertEqual(parts[:2], (4, 4))
        self.assertIs(parts[4], bm.full_kv_num_blocks)
        self.assertIs(parts[8], bm.full_q_num_blocks)
        self.assertIsNone(parts[10])
        self.assertIsNone(parts[12])
        self.assertIsNone(parts[13])
        self.assertEqual(parts[14:16], (2, 2))

    def test_as_tuple_distinguishes_different_masks(self):
        # Two masks with different kv layout must not produce equal tuples.
        dense = self._build_mask(self.device_name)
        kv_num = torch.tensor([[[2, 1]]], device=self.device_name)
        kv_idx = torch.tensor([[[[0, 1], [2, 0]]]], device=self.device_name)
        sparse = BlockMask.from_kv_blocks(kv_num, kv_idx, None, None,
                                          BLOCK_SIZE=(2, 2), seq_lengths=(4, 4))
        parts_dense = dense.as_tuple()
        parts_sparse = sparse.as_tuple()
        self.assertFalse(bool(torch.equal(parts_dense[2].cpu(), parts_sparse[2].cpu())))


if __name__ == "__main__":
    run_tests()
