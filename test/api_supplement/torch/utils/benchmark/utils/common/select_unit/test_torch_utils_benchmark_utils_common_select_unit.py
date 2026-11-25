# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.benchmark.utils.common.select_unit 接口功能正确性
API 名称：torch.utils.benchmark.utils.common.select_unit
API 签名：torch.utils.benchmark.utils.common.select_unit(t)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空容器输入                     | 未覆盖（见下）                                     |
| 枚举选项         | 返回时间单位档位（ns/us/ms/s）       | 已覆盖（nanoseconds、microseconds、milliseconds、seconds） |
| 参数类型         | float、int 及非法 str 输入           | 已覆盖（seconds、invalid_type_raises）             |
| 传参与不传参     | 必选参数传入与缺参                   | 已覆盖（seconds、missing_arg_raises）              |
| 等价类/边界值    | 10 的幂次阈值、亚纳秒及 0.0 边界     | 已覆盖（threshold_boundaries、zero_raises）        |
| 正常传参场景     | 各量级正数时间映射至对应单位与倍率   | 已覆盖（nanoseconds、microseconds、milliseconds、seconds） |
| 异常传参场景     | 0.0、缺参及非数值字符串触发异常      | 已覆盖（zero_raises、missing_arg_raises、invalid_type_raises） |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 空/非空：接口仅接收单个标量时间值，无容器或空值输入维度。
- 混合设备类型：纯 Python 数值格式化辅助函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.benchmark.utils.common import select_unit

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestSelectUnit(TestCase):
    """Test cases for torch.utils.benchmark.utils.common.select_unit."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_nanoseconds(self):
        """Values in [1e-9, 1e-6) map to ('ns', 1e-9)."""
        unit, scale = select_unit(1e-8)
        self.assertEqual(unit, "ns")
        self.assertEqual(scale, 1e-9)
        self.assertAlmostEqual(1e-8 / scale, 10.0, places=5)

    def test_microseconds(self):
        """Values in [1e-6, 1e-3) map to ('us', 1e-6)."""
        unit, scale = select_unit(1e-5)
        self.assertEqual(unit, "us")
        self.assertEqual(scale, 1e-6)
        self.assertAlmostEqual(1e-5 / scale, 10.0, places=5)

    def test_milliseconds(self):
        """Values in [1e-3, 1.0) map to ('ms', 1e-3)."""
        unit, scale = select_unit(1e-2)
        self.assertEqual(unit, "ms")
        self.assertEqual(scale, 1e-3)
        self.assertAlmostEqual(1e-2 / scale, 10.0, places=5)

    def test_seconds(self):
        """Values >= 1.0 map to ('s', 1)."""
        unit, scale = select_unit(1.5)
        self.assertEqual(unit, "s")
        self.assertEqual(scale, 1)
        unit_int, scale_int = select_unit(10)
        self.assertEqual((unit_int, scale_int), ("s", 1))

    def test_threshold_boundaries(self):
        """Exact powers of ten and sub-nanosecond values select expected units."""
        self.assertEqual(select_unit(1e-9), ("ns", 1e-9))
        self.assertEqual(select_unit(1e-6), ("us", 1e-6))
        self.assertEqual(select_unit(1e-3), ("ms", 1e-3))
        self.assertEqual(select_unit(1.0), ("s", 1))
        self.assertEqual(select_unit(1e-12), ("s", 1))

    def test_zero_raises(self):
        """Passing 0.0 causes log10(0) = -inf and raises OverflowError or ValueError."""
        with self.assertRaises((OverflowError, ValueError)):
            select_unit(0.0)

    def test_missing_arg_raises(self):
        """Calling select_unit with no arguments raises TypeError."""
        with self.assertRaises(TypeError):
            select_unit()

    def test_invalid_type_raises(self):
        """Passing a non-numeric string raises TypeError or ValueError."""
        with self.assertRaises((TypeError, ValueError)):
            select_unit("1ms")


if __name__ == "__main__":
    run_tests()
