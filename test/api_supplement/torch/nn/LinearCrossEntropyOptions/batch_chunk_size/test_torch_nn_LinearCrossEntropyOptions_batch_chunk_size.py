# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LinearCrossEntropyOptions.batch_chunk_size 接口功能正确性
API 名称：torch.nn.LinearCrossEntropyOptions.batch_chunk_size
API 签名：batch_chunk_size: int | None = None（dataclass 字段，构造参数 batch_chunk_size=...）

说明：batch_chunk_size 是 LinearCrossEntropyOptions 的批分块大小字段：每个 chunk 处理的
batch 行数，None（默认）表示整个 batch 一个 chunk；必须为正整数或 None，非法值在 dataclass
__post_init__ 校验即抛 ValueError。与 chunking_method 同时设置且解析结果不一致时，
_adjust 抛 ValueError 提示只传其一。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认 None 与显式正整数 | 已覆盖（_default_none/_valid_values） |
| 枚举选项 | 正整数（1、3）与 None | 已覆盖（_valid_values） |
| 参数类型 | int 字段保持；float/零/负数被拒 | 已覆盖（_valid_values/_invalid_raises） |
| 传参与不传参 | 省略（默认 None）与显式传参 | 已覆盖（_default_none/_valid_values） |
| 等价类/边界值 | _adjust 将显式值截断到 num_batches；与 auto 组合关闭启发式 | 已覆盖（_adjust_clamps_and_disables_auto） |
| 正常传参场景 | _adjust 解析后 batch_chunk_size 契约 | 已覆盖（_adjust_clamps_and_disables_auto） |
| 异常传参场景 | 0/负数/浮点构造即抛 ValueError；与 chunking_method 冲突解析即抛 | 已覆盖（_invalid_raises/_conflict_raises） |
| 混合设备类型 | 纯 Python 分块参数，无设备执行路径 | 未覆盖（解析不依赖设备类型，见下） |

未覆盖项及原因：
- 混合设备类型：分块大小解析不依赖设备类型；实际分块执行效果由
  F.linear_cross_entropy 功能用例（chunked batch_chunk_size 场景）在设备上验证。
- 解析规则以 PyTorch 2.15.0 的 _adjust 实现为准（内部 API，上游测试亦按该调用点使用），
  跨版本行为可能变化。

注意：纯 Python 字段契约按取值、校验与解析结果断言；无浮点 Tensor 输出，不做
     CPU Golden 数值对照。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Pure-container API: device probing stays opportunistic and never raises.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLinearCrossEntropyOptionsBatchChunkSize(TestCase):
    """Test cases for the batch_chunk_size field of LinearCrossEntropyOptions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_batch_chunk_size_default_none(self):
        """Omitted batch_chunk_size resolves to the documented None default."""
        self.assertIsNone(torch.nn.LinearCrossEntropyOptions().batch_chunk_size)

    def test_batch_chunk_size_valid_values(self):
        """Positive chunk sizes are stored verbatim."""
        for size in (1, 3, 1024):
            options = torch.nn.LinearCrossEntropyOptions(batch_chunk_size=size)
            self.assertEqual(options.batch_chunk_size, size)

    def test_batch_chunk_size_invalid_raises(self):
        """Non-positive or non-int sizes are rejected at construction."""
        for size in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "batch_chunk_size"):
                torch.nn.LinearCrossEntropyOptions(batch_chunk_size=size)

    def test_batch_chunk_size_adjust_clamps_and_disables_auto(self):
        """_adjust clamps to num_batches and disables the auto heuristic."""
        resolved = torch.nn.LinearCrossEntropyOptions(batch_chunk_size=16)._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=None,
        )
        # The chunk size is clamped to the batch size.
        self.assertEqual(resolved.batch_chunk_size, 8)
        # An explicit size disables the chunking_method='auto' heuristic.
        self.assertIsNone(resolved.chunking_method)
        resolved = torch.nn.LinearCrossEntropyOptions(batch_chunk_size=3)._adjust(
            num_batches=8,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=None,
        )
        self.assertEqual(resolved.batch_chunk_size, 3)
        self.assertIsNone(resolved.chunking_method)

    def test_batch_chunk_size_conflict_raises(self):
        """A conflicting chunking_method raises ValueError at resolution time."""
        with self.assertRaisesRegex(ValueError, "pass only one"):
            torch.nn.LinearCrossEntropyOptions(
                batch_chunk_size=2, chunking_method="aspect_ratio"
            )._adjust(
                num_batches=8,
                in_features=4,
                num_classes=10,
                dtype=torch.float32,
                device=None,
            )


if __name__ == "__main__":
    run_tests()
