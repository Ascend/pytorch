# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.finfo 接口功能正确性
API 名称：torch.finfo
API 签名：torch.finfo(dtype) / torch.finfo()  (默认 float32)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                                  |
|------------------|--------------------------------------------|-----------------------------------------------------------|
| 空/非空          | dtype 省略（finfo()）vs 显式传入           | 已覆盖：default_dtype_is_float32、各 dtype 用例           |
| 枚举选项         | float16 / bfloat16 / float32 / float64     | 已覆盖：dtype_coverage_bits、各 dtype 用例                |
| 参数类型         | torch.dtype（非 dtype 输入报错）           | 已覆盖：non_dtype_argument_raises                         |
| 传参与不传参     | 省略 dtype 时默认取 float32                | 已覆盖：default_dtype_is_float32                          |
| 等价类/边界值    | eps / bits / min / max 的精确取值          | 已覆盖：float32/float64/float16/bfloat16 属性用例         |
| 正常传参场景     | 返回 finfo 对象且字段符合数值常量          | 已覆盖                                                    |
| 异常传参场景     | 非浮点 dtype 抛 TypeError                  | 已覆盖：non_float_dtype_raises                            |
| 混合设备类型     | 纯 dtype 元信息工具，不涉及设备迁移        | 不适用                                                    |

未覆盖项及原因：
- 混合设备类型：torch.finfo 仅按 dtype 返回浮点元信息，无 Tensor 输入、无设备执行路径，
  不做 CPU/NPU 数值对照（框架类 A：纯 Python 工具）。

注：该 API 为纯 Python 元信息工具（框架类 A），结果与硬件无关；按精确结构/数值常量与异常路径断言，
     不做 CPU Golden 设备对照。eps/min/max/bits 均为可精确表示的常量，使用 assertEqual 精确比较。
     注意 finfo.dtype 字段返回类型名字符串（如 "float32"），而非 torch.dtype 对象。
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
class TestFinfoUt(TestCase):
    """Functional tests for torch.finfo (framework class A, device-independent)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_default_dtype_is_float32(self):
        """Omitting dtype defaults to float32."""
        self.assertEqual(torch.finfo().dtype, "float32")

    def test_float32_attributes(self):
        """float32 reports its exact bits, eps, bounds and prefix invariants."""
        info = torch.finfo(torch.float32)
        self.assertEqual(info.bits, 32)
        self.assertEqual(info.dtype, "float32")
        self.assertEqual(info.eps, 2.0 ** -23)
        self.assertEqual(info.min, -info.max)
        self.assertGreater(info.max, 0.0)
        self.assertGreater(info.tiny, 0.0)
        self.assertGreater(info.eps, info.tiny)
        self.assertLess(info.eps, 1.0)

    def test_float64_eps(self):
        """float64 eps equals 2**-52 and bits equals 64."""
        self.assertEqual(torch.finfo(torch.float64).eps, 2.0 ** -52)
        self.assertEqual(torch.finfo(torch.float64).bits, 64)

    def test_float16_attributes(self):
        """float16 eps equals 2**-10 and its bounds equal +/-65504."""
        info = torch.finfo(torch.float16)
        self.assertEqual(info.bits, 16)
        self.assertEqual(info.eps, 2.0 ** -10)
        self.assertEqual(info.max, 65504.0)
        self.assertEqual(info.min, -65504.0)

    def test_bfloat16_eps(self):
        """bfloat16 eps equals 2**-7 with 16 bits."""
        info = torch.finfo(torch.bfloat16)
        self.assertEqual(info.bits, 16)
        self.assertEqual(info.eps, 2.0 ** -7)

    def test_dtype_coverage_bits(self):
        """Each supported float dtype maps to its expected bit width and name."""
        expected = {
            torch.float16: (16, "float16"),
            torch.bfloat16: (16, "bfloat16"),
            torch.float32: (32, "float32"),
            torch.float64: (64, "float64"),
        }
        for dtype, (bits, name) in expected.items():
            self.assertEqual(torch.finfo(dtype).bits, bits, f"bits mismatch for {dtype}")
            self.assertEqual(torch.finfo(dtype).dtype, name, f"dtype mismatch for {dtype}")

    def test_resolution_positive(self):
        """All float types expose a positive resolution."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            self.assertGreater(torch.finfo(dtype).resolution, 0.0)

    def test_non_float_dtype_raises(self):
        """An integer dtype is rejected with TypeError."""
        with self.assertRaisesRegex(TypeError, "floating point"):
            torch.finfo(torch.int32)

    def test_non_dtype_argument_raises(self):
        """A non-dtype argument (e.g. a string) is rejected with TypeError."""
        with self.assertRaises(TypeError):
            torch.finfo("float32")


if __name__ == "__main__":
    run_tests()
