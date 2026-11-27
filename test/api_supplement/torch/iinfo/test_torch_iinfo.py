# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.iinfo 接口功能正确性
API 名称：torch.iinfo
API 签名：torch.iinfo(dtype)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                                  |
|------------------|--------------------------------------------|-----------------------------------------------------------|
| 空/非空          | dtype 必填，省略时报错                     | 已覆盖：missing_argument_raises                           |
| 枚举选项         | int8/int16/int32/int64/uint8               | 已覆盖：各 dtype 属性用例、range_count_matches_bits      |
| 参数类型         | torch.dtype（非 dtype / bool 输入报错）    | 已覆盖：bool_dtype_raises                                |
| 传参与不传参     | dtype 为必填参数                           | 已覆盖：missing_argument_raises                           |
| 等价类/边界值    | min / max / bits 的精确取值                | 已覆盖：各 dtype 属性用例                                 |
| 正常传参场景     | 返回 iinfo 对象且字段符合整数常量          | 已覆盖                                                    |
| 异常传参场景     | 浮点 / 布尔 dtype 抛 TypeError             | 已覆盖：float_dtype_raises、bool_dtype_raises             |
| 混合设备类型     | 纯 dtype 元信息工具，不涉及设备迁移        | 不适用                                                    |

未覆盖项及原因：
- 混合设备类型：torch.iinfo 仅按 dtype 返回整数元信息，无 Tensor 输入、无设备执行路径，
  不做 CPU/NPU 数值对照（框架类 A：纯 Python 工具）。

注：该 API 为纯 Python 元信息工具（框架类 A），结果与硬件无关；按精确结构/数值常量与异常路径断言，
     不做 CPU Golden 设备对照。bits/min/max 均为精确整数常量，使用 assertEqual 精确比较。
     注意 iinfo.dtype 字段返回类型名字符串（如 "int8"），而非 torch.dtype 对象。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestIinfoUt(TestCase):
    """Functional tests for torch.iinfo (framework class A, device-independent)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_int8_attributes(self):
        """int8 spans [-128, 127] with 8 bits."""
        info = torch.iinfo(torch.int8)
        self.assertEqual(info.bits, 8)
        self.assertEqual(info.min, -128)
        self.assertEqual(info.max, 127)
        self.assertEqual(info.dtype, "int8")

    def test_int16_attributes(self):
        """int16 spans [-32768, 32767] with 16 bits."""
        info = torch.iinfo(torch.int16)
        self.assertEqual(info.bits, 16)
        self.assertEqual(info.min, -32768)
        self.assertEqual(info.max, 32767)

    def test_int32_attributes(self):
        """int32 spans [-2**31, 2**31 - 1] with 32 bits."""
        info = torch.iinfo(torch.int32)
        self.assertEqual(info.bits, 32)
        self.assertEqual(info.min, -2 ** 31)
        self.assertEqual(info.max, 2 ** 31 - 1)

    def test_int64_attributes(self):
        """int64 spans [-2**63, 2**63 - 1] with 64 bits."""
        info = torch.iinfo(torch.int64)
        self.assertEqual(info.bits, 64)
        self.assertEqual(info.min, -2 ** 63)
        self.assertEqual(info.max, 2 ** 63 - 1)

    def test_uint8_attributes(self):
        """uint8 spans [0, 255] with 8 bits."""
        info = torch.iinfo(torch.uint8)
        self.assertEqual(info.bits, 8)
        self.assertEqual(info.min, 0)
        self.assertEqual(info.max, 255)
        self.assertEqual(info.dtype, "uint8")

    def test_range_count_matches_bits(self):
        """For every integer dtype, the representable count equals 2**bits."""
        for dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            info = torch.iinfo(dtype)
            self.assertEqual(info.max - info.min + 1, 2 ** info.bits, f"range mismatch for {dtype}")

    def test_float_dtype_raises(self):
        """A floating dtype is rejected with TypeError."""
        with self.assertRaisesRegex(TypeError, "integer"):
            torch.iinfo(torch.float32)

    def test_bool_dtype_raises(self):
        """torch.bool is not supported by iinfo and raises TypeError."""
        with self.assertRaises(TypeError):
            torch.iinfo(torch.bool)

    def test_missing_argument_raises(self):
        """dtype is mandatory; calling without it raises TypeError."""
        with self.assertRaises(TypeError):
            torch.iinfo()


if __name__ == "__main__":
    run_tests()
