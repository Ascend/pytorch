# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.attention.flex_attention.BlockMask.to_string 接口功能正确性
API 名称：torch.nn.attention.flex_attention.BlockMask.to_string
API 签名：to_string(self, grid_size: int | tuple[int, int] = (20, 20), limit: int = 4) -> str

覆盖维度表：
| 覆盖维度         | 说明                                                | 覆盖情况                       |
|------------------|-----------------------------------------------------|--------------------------------|
| 参数类型         | 返回 str                                            | 已覆盖                         |
| 枚举选项         | grid_size int/tuple/-1 三种形态                     | 已覆盖                         |
| 等价类/边界值    | grid_size=-1 全量打印与 int 压缩打印                | 已覆盖                         |
| 传参与不传参     | 默认参数与显式 grid_size/limit                      | 已覆盖                         |
| 正常传参场景     | 已知布局的精确字符金标准                            | 已覆盖                         |
| 异常传参场景     | 不适用：越界 grid_size 按步长裁剪，无稳定异常       | 不适用                         |
| 混合设备类型     | 不适用：单实例方法，无多 Tensor 入参                | 不适用                         |
| 副作用验证       | limit 截断输出省略号与提示行                        | 已覆盖                         |

未覆盖项及原因：
- 无

注意：输出为可视化字符串，按手工推导的精确字符序列断言（含 █/░/空格），
     并与 CPU 构建的同构掩码输出字符串全等对照。
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


from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402


def causal_mask(b, h, q_idx, kv_idx):
    # 64x64 with 32-blocks gives a 2x2 block grid [[1,0],[1,1]].
    return q_idx >= kv_idx


CAUSAL_2X2_FULL = "(0, 0)\n██  \n████\n"
CAUSAL_2X2_COMPRESSED = "(0, 0)\n░░\n"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBlockMaskToString(TestCase):
    """Test BlockMask.to_string visualization contracts (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)
        self.bm = create_block_mask(causal_mask, 1, 1, 64, 64, device=self.device_name,
                                    BLOCK_SIZE=(32, 32))

    def test_to_string_returns_str(self):
        # The API must return a plain Python string for the default arguments.
        rendered = self.bm.to_string()
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 0)

    def test_to_string_uncompressed_exact_golden(self):
        # grid_size=-1 prints every block row: exact known visualization.
        self.assertEqual(self.bm.to_string(grid_size=-1), CAUSAL_2X2_FULL)

    def test_to_string_compressed_grid_exact_golden(self):
        # grid_size=1 collapses the 2x2 grid into one averaged cell (density .75).
        self.assertEqual(self.bm.to_string(grid_size=1), CAUSAL_2X2_COMPRESSED)

    def test_to_string_matches_cpu_golden(self):
        # Visualization is device independent: device and CPU builds render equal.
        bm_cpu = create_block_mask(causal_mask, 1, 1, 64, 64, device="cpu",
                                   BLOCK_SIZE=(32, 32))
        self.assertEqual(self.bm.to_string(grid_size=-1), bm_cpu.to_string(grid_size=-1))
        self.assertEqual(self.bm.to_string(grid_size=(2, 2)),
                         bm_cpu.to_string(grid_size=(2, 2)))

    def test_to_string_tuple_grid_limits_rows(self):
        # grid_size=(2, 2) on a 4x4 grid keeps two rows; -1 keeps all four.
        big = create_block_mask(causal_mask, 1, 1, 128, 128, device=self.device_name,
                                BLOCK_SIZE=(32, 32))
        compressed = big.to_string(grid_size=(2, 2))
        full = big.to_string(grid_size=-1)
        self.assertEqual(compressed.count("\n"), 3)
        self.assertEqual(full.count("\n"), 5)
        self.assertLess(len(compressed), len(full))

    def test_to_string_limit_truncates_batch_sections(self):
        # With B*H sections above limit, output ends with ellipsis and a hint.
        big = create_block_mask(causal_mask, 2, 2, 64, 64, device=self.device_name,
                                BLOCK_SIZE=(32, 32))
        truncated = big.to_string(limit=2)
        self.assertIn("...", truncated)
        self.assertIn("set BlockMask.to_string(limit=N)", truncated)
        complete = big.to_string(limit=10)
        self.assertNotIn("...", complete)
        self.assertNotIn("set BlockMask.to_string(limit=N)", complete)

    def test_to_string_repeated_calls_deterministic(self):
        # Rendering the same mask twice must be byte-identical.
        self.assertEqual(self.bm.to_string(grid_size=-1), self.bm.to_string(grid_size=-1))


if __name__ == "__main__":
    run_tests()
