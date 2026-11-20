# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LinearCrossEntropyOptions 接口功能正确性
API 名称：torch.nn.LinearCrossEntropyOptions
API 签名：LinearCrossEntropyOptions(allow_retain_graph=False, batch_chunk_size=None,
chunking_method='auto', acc_policy='auto', acc_dtype=None)

说明：torch.nn.LinearCrossEntropyOptions 是 torch.nn.modules.linear_cross_entropy_options
模块导出的 frozen slots dataclass，作为 F.linear_cross_entropy 的分块计算选项容器；
5 个字段（acc_dtype/acc_policy/allow_retain_graph/batch_chunk_size/chunking_method）为
dataclass member，字段级行为由各自独立文件看护。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 零参构造与全字段显式构造 | 已覆盖（_default_fields/_keyword_construction） |
| 枚举选项 | 各字段合法值组合构造 | 已覆盖（_keyword_construction） |
| 参数类型 | bool/int/None/str/dtype 字段类型保持 | 已覆盖（_keyword_construction/_default_fields） |
| 传参与不传参 | 默认值契约与关键字传参 | 已覆盖（_default_fields/_keyword_construction） |
| 等价类/边界值 | frozen 不可变、slots 无 __dict__、replace 派生新实例 | 已覆盖（_dataclass_contract/_replace） |
| 正常传参场景 | 与 torch.nn.modules 命名空间同一对象、dataclass fields 元数据 | 已覆盖（_module_identity/_dataclass_contract） |
| 异常传参场景 | 非法 acc_policy/chunking_method/batch_chunk_size/acc_dtype 构造即抛 ValueError | 已覆盖（_invalid_*_raises） |
| 混合设备类型 | 纯 Python 容器契约，构造与校验不依赖设备 | 未覆盖（本 API 全部测试均无设备执行路径，见下） |

未覆盖项及原因：
- 混合设备类型：本 API 是纯 Python 选项容器，构造、默认值与校验均不产生设备执行路径，
  无可对照的设备差异契约。
- 字段在分块计算中的解析行为（auto 解析、冲突检测）由 F.linear_cross_entropy 功能用例及
  各字段独立文件看护，本文件仅看护容器契约。

注意：纯 Python 工具按返回值、结构与异常路径断言；本 API 无浮点 Tensor 输出，不做
     CPU Golden 数值对照。约束：字段校验行为以 PyTorch 2.15.0 源码为准。
"""
import dataclasses
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
class TestLinearCrossEntropyOptions(TestCase):
    """Test cases for torch.nn.LinearCrossEntropyOptions (device independent)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_linear_cross_entropy_options_default_fields(self):
        """Zero-argument construction keeps the documented defaults."""
        options = torch.nn.LinearCrossEntropyOptions()
        self.assertFalse(options.allow_retain_graph)
        self.assertIsNone(options.batch_chunk_size)
        self.assertEqual(options.chunking_method, "auto")
        self.assertEqual(options.acc_policy, "auto")
        self.assertIsNone(options.acc_dtype)

    def test_linear_cross_entropy_options_keyword_construction(self):
        """Explicit keyword values are stored per field."""
        options = torch.nn.LinearCrossEntropyOptions(
            allow_retain_graph=True,
            batch_chunk_size=7,
            chunking_method="aspect_ratio:2",
            acc_policy="accurate",
            acc_dtype=torch.float32,
        )
        self.assertTrue(options.allow_retain_graph)
        self.assertEqual(options.batch_chunk_size, 7)
        self.assertEqual(options.chunking_method, "aspect_ratio:2")
        self.assertEqual(options.acc_policy, "accurate")
        self.assertIs(options.acc_dtype, torch.float32)

    def test_linear_cross_entropy_options_module_identity(self):
        """torch.nn re-exports the modules.linear_cross_entropy_options class."""
        self.assertIs(
            torch.nn.LinearCrossEntropyOptions,
            torch.nn.modules.linear_cross_entropy_options.LinearCrossEntropyOptions,
        )

    def test_linear_cross_entropy_options_dataclass_contract(self):
        """Frozen slots dataclass: ordered fields, no mutation, no __dict__."""
        options = torch.nn.LinearCrossEntropyOptions()
        field_names = [field.name for field in dataclasses.fields(options)]
        self.assertEqual(
            field_names,
            [
                "allow_retain_graph",
                "batch_chunk_size",
                "chunking_method",
                "acc_policy",
                "acc_dtype",
            ],
        )
        self.assertFalse(hasattr(options, "__dict__"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            options.acc_policy = "compact"
        # slots=True rejects attributes outside the declared fields. The
        # raised type varies across stdlib versions: on frozen+slots
        # dataclasses of CPython 3.11/3.12 (this environment) undeclared
        # names fall through the frozen guard and raise TypeError from the
        # slots rewrite's super() call, while other versions raise
        # FrozenInstanceError (an AttributeError subclass). Accept both;
        # assignment must be rejected either way.
        with self.assertRaises((TypeError, AttributeError)):
            options.extra_attribute = "unused"

    def test_linear_cross_entropy_options_replace(self):
        """dataclasses.replace derives a new instance and keeps the original."""
        original = torch.nn.LinearCrossEntropyOptions(acc_policy="accurate")
        derived = dataclasses.replace(original, batch_chunk_size=3)
        self.assertEqual(derived.batch_chunk_size, 3)
        self.assertEqual(derived.acc_policy, "accurate")
        self.assertIsNone(original.batch_chunk_size)
        self.assertIsNot(original, derived)

    def test_linear_cross_entropy_options_invalid_acc_policy_raises(self):
        """Unknown acc_policy values are rejected at construction."""
        with self.assertRaisesRegex(ValueError, "acc_policy"):
            torch.nn.LinearCrossEntropyOptions(acc_policy="fast")
        with self.assertRaisesRegex(ValueError, "acc_policy"):
            torch.nn.LinearCrossEntropyOptions(acc_policy=1)

    def test_linear_cross_entropy_options_invalid_chunking_method_raises(self):
        """Unknown chunking_method values are rejected at construction."""
        for method in ("row", "aspect_ratio:0", "aspect_ratio:-1", "aspect_ratio:x"):
            with self.assertRaisesRegex(ValueError, "chunking_method"):
                torch.nn.LinearCrossEntropyOptions(chunking_method=method)

    def test_linear_cross_entropy_options_invalid_batch_chunk_size_raises(self):
        """batch_chunk_size must be a positive int or None."""
        for size in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "batch_chunk_size"):
                torch.nn.LinearCrossEntropyOptions(batch_chunk_size=size)

    def test_linear_cross_entropy_options_invalid_acc_dtype_raises(self):
        """acc_dtype must be None or one of fp16/bf16/fp32/fp64."""
        with self.assertRaisesRegex(ValueError, "acc_dtype"):
            torch.nn.LinearCrossEntropyOptions(acc_dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "acc_dtype"):
            torch.nn.LinearCrossEntropyOptions(acc_dtype=torch.complex64)


if __name__ == "__main__":
    run_tests()
