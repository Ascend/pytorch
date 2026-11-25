# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.throughput_benchmark.format_time 接口功能正确性
API 名称：torch.utils.throughput_benchmark.format_time
API 签名：torch.utils.throughput_benchmark.format_time(time_us=None, time_ms=None, time_s=None) -> str

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 全 None（不传值）与单单位非空数值    | 已覆盖（no_args_raises、format_microseconds）      |
| 枚举选项         | time_us / time_ms / time_s 三种单位  | 已覆盖（format_microseconds、format_milliseconds、 |
|                  |                                      | format_seconds）                                   |
| 参数类型         | int 与 float 数值参数                | 已覆盖（format_microseconds、format_milliseconds、 |
|                  |                                      | format_seconds）                                   |
| 传参与不传参     | 无参、单参、多参同时传入             | 已覆盖（no_args_raises、format_microseconds、      |
|                  |                                      | multiple_args_raises）                             |
| 等价类/边界值    | 0 值及 1000us/1000ms/0.5s 单位换算   | 已覆盖（format_microseconds、                      |
|                  |                                      | boundary_unit_promotion）                          |
| 正常传参场景     | 微秒、毫秒、秒格式化为三位小数字符串 | 已覆盖（format_microseconds、format_milliseconds、 |
|                  |                                      | format_seconds、boundary_unit_promotion）          |
| 异常传参场景     | 无参、多单位并传与未知关键字参数异常 | 已覆盖（no_args_raises、multiple_args_raises、     |
|                  |                                      | unexpected_kwarg_raises）                          |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯标量时间格式化工具，无 Tensor 输入）     |

未覆盖项及原因：
- 混合设备类型：该接口仅对标量时间数值做单位换算与字符串格式化，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.throughput_benchmark import format_time

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
class TestFormatTime(TestCase):
    """Test cases for torch.utils.throughput_benchmark.format_time."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_format_microseconds(self):
        """time_us below 1000 formats with 'us' suffix and 3 decimal places."""
        self.assertEqual(format_time(time_us=500), "500.000us")
        self.assertEqual(format_time(time_us=0), "0.000us")

    def test_format_milliseconds(self):
        """time_ms in [1, 1000) formats with 'ms' suffix and 3 decimal places."""
        self.assertEqual(format_time(time_ms=12.5), "12.500ms")

    def test_format_seconds(self):
        """time_s >= 1 formats with 's' suffix and 3 decimal places."""
        self.assertEqual(format_time(time_s=2.0), "2.000s")

    def test_boundary_unit_promotion(self):
        """Values crossing 1000us or 1e6us promote to ms and s respectively."""
        self.assertEqual(format_time(time_us=1000), "1.000ms")
        self.assertEqual(format_time(time_ms=1000), "1.000s")
        self.assertEqual(format_time(time_s=0.5), "500.000ms")

    def test_no_args_raises(self):
        """Calling format_time with no time arguments raises AssertionError."""
        with self.assertRaises((AssertionError, ValueError)):
            format_time()

    def test_multiple_args_raises(self):
        """Providing more than one time unit argument raises AssertionError."""
        with self.assertRaises((AssertionError, ValueError)):
            format_time(time_us=10, time_ms=1)
        with self.assertRaises((AssertionError, ValueError)):
            format_time(time_us=1, time_ms=1, time_s=1)

    def test_unexpected_kwarg_raises(self):
        """Passing an unknown keyword argument raises TypeError."""
        with self.assertRaises(TypeError):
            format_time(time_ns=100)


if __name__ == "__main__":
    run_tests()
