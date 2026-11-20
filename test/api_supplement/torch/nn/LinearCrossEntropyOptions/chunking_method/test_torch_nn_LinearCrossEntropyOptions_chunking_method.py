# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LinearCrossEntropyOptions.chunking_method 接口功能正确性
API 名称：torch.nn.LinearCrossEntropyOptions.chunking_method
API 签名：chunking_method: str | None = 'auto'（dataclass 字段，构造参数 chunking_method=...）

说明：chunking_method 是 LinearCrossEntropyOptions 的分块启发式字段：'auto'（默认）解析为
'aspect_ratio'；'aspect_ratio' 按 next_pow2(ceil(num_batches / ceil(num_classes /
in_features))) 选取每 chunk 行数；'aspect_ratio:N'（N>=1）再除以 N；None 关闭启发式、
使用 batch_chunk_size。非法取值在 __post_init__ 校验即抛 ValueError。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 默认 'auto'、None 关闭启发式与显式方法串 | 已覆盖（_default_auto/_valid_values） |
| 枚举选项 | None/'auto'/'aspect_ratio'/'aspect_ratio:N' 全部合法值 | 已覆盖（_valid_values） |
| 参数类型 | 字符串方法串与 None，非法字符串被拒 | 已覆盖（_valid_values/_invalid_raises） |
| 传参与不传参 | 省略（默认 'auto'）与显式传参 | 已覆盖（_default_auto/_valid_values） |
| 等价类/边界值 | 因子 1/2/3、num_batches=1、7/2/8 非整除组合 | 已覆盖（_compute_batch_chunk_size） |
| 正常传参场景 | 'auto' 经 _adjust 解析为 'aspect_ratio' 并按公式给 chunk | 已覆盖（_auto_resolution） |
| 异常传参场景 | 非法方法串（非 aspect_ratio、因子 0/负数/非数字）构造即抛 | 已覆盖（_invalid_raises） |
| 混合设备类型 | 分块启发式为纯 Python 公式，无设备执行路径 | 未覆盖（解析不依赖设备类型，见下） |

未覆盖项及原因：
- 混合设备类型：启发式公式不依赖设备类型；实际分块执行效果由 F.linear_cross_entropy
  功能用例（chunked chunking_method 场景）在设备上验证。
- 'auto' 的 compact 路径 per-target B_ref 内存上限仅以实现内部细节存在，未断言具体上限值；
  解析断言限定在文档化的 aspect_ratio 公式与 'auto'→'aspect_ratio' 改写。
- 解析规则以 PyTorch 2.15.0 的 _compute_batch_chunk_size/_adjust 实现为准（内部 API，
  上游测试亦按该调用点使用），跨版本行为可能变化。

注意：纯 Python 字段契约按取值、校验与公式结果断言；无浮点 Tensor 输出，不做
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
class TestLinearCrossEntropyOptionsChunkingMethod(TestCase):
    """Test cases for the chunking_method field of LinearCrossEntropyOptions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.options = torch.nn.LinearCrossEntropyOptions()

    def test_chunking_method_default_auto(self):
        """Omitted chunking_method resolves to the documented 'auto' default."""
        self.assertEqual(self.options.chunking_method, "auto")

    def test_chunking_method_valid_values(self):
        """All documented method forms are stored verbatim."""
        for method in (None, "auto", "aspect_ratio", "aspect_ratio:4"):
            options = torch.nn.LinearCrossEntropyOptions(chunking_method=method)
            self.assertEqual(options.chunking_method, method)

    def test_chunking_method_invalid_raises(self):
        """Unknown methods and bad factors are rejected at construction."""
        for method in (
            "row",
            "aspect_ratio:0",
            "aspect_ratio:-1",
            "aspect_ratio:x",
            "aspect_ratio:",
        ):
            with self.assertRaisesRegex(ValueError, "chunking_method"):
                torch.nn.LinearCrossEntropyOptions(chunking_method=method)

    def test_chunking_method_compute_batch_chunk_size(self):
        """The documented aspect_ratio formula holds, including the factor split."""
        compute = self.options._compute_batch_chunk_size
        # ceil(10/4)=3 chunk-target ceil(8/3)=3 -> next pow2 4.
        self.assertEqual(compute(8, 4, 10, "aspect_ratio"), 4)
        # Factor 2 and 3 divide the pow2 chunk size.
        self.assertEqual(compute(8, 4, 10, "aspect_ratio:2"), 2)
        self.assertEqual(compute(8, 4, 10, "aspect_ratio:3"), 1)
        # Single batch keeps a single chunk.
        self.assertEqual(compute(1, 4, 10, "aspect_ratio"), 1)
        # ceil(16/3)=6 -> next pow2 8.
        self.assertEqual(compute(16, 4, 10, "aspect_ratio"), 8)
        # ceil(8/2)=4 chunk-target ceil(7/4)=2 -> next pow2 2.
        self.assertEqual(compute(7, 2, 8, "aspect_ratio"), 2)

    def test_chunking_method_auto_resolution(self):
        """_adjust rewrites 'auto' to 'aspect_ratio' with the formula chunk."""
        resolved = self.options._adjust(
            num_batches=16,
            in_features=4,
            num_classes=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(resolved.chunking_method, "aspect_ratio")
        self.assertEqual(resolved.batch_chunk_size, 8)


if __name__ == "__main__":
    run_tests()
