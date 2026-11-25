# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.FormattedTimesMixin 接口功能正确性
API 名称：torch.autograd.profiler_util.FormattedTimesMixin
API 签名：class torch.autograd.profiler_util.FormattedTimesMixin

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 裸实例缺属性与完整子类实例属性读取 | 已覆盖（missing_attributes_raise、nonzero_count_exact_division） |
| 枚举选项 | 无枚举参数 | 未覆盖（纯属性 Mixin 类，无枚举选项） |
| 参数类型 | 无参构造与传入多余参数报错 | 已覆盖（missing_attributes_raise） |
| 传参与不传参 | 无参构造与带参构造报错 | 已覆盖（missing_attributes_raise） |
| 等价类/边界值 | count=0 防除零、us/ms/s 跨量级格式化 | 已覆盖（count_zero_returns_zero、formatted_strings_across_units） |
| 正常传参场景 | 均值、格式化与警告 | 已覆盖（count_zero_returns_zero、nonzero_count_exact_division、formatted_strings_across_units 等） |
| 异常传参场景 | 缺少时间属性 AttributeError、多参 TypeError | 已覆盖（missing_attributes_raise） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 计时格式化 Mixin，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口为属性计算 Mixin 类，不接收枚举参数。
- 混合设备类型：接口仅对标量计时字段做除法与单位格式化，不接收 Tensor 参数。

注意：本接口为计时属性格式化 Mixin 类，无浮点 Tensor 输出，
     按 API 语义使用数值精度、格式化字符串、告警及异常断言验证。
"""
import os
import unittest
import warnings

import torch
from torch.autograd.profiler_util import FormattedTimesMixin

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

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


class _DummyTimedEvent(FormattedTimesMixin):

    def __init__(self, count, cpu=0.0, dev=0.0, self_cpu=0.0, self_dev=0.0):
        self.count = count
        self.cpu_time_total = cpu
        self.device_time_total = dev
        self.self_cpu_time_total = self_cpu
        self.self_device_time_total = self_dev


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFormattedTimesMixin(TestCase):
    """Test cases for torch.autograd.profiler_util.FormattedTimesMixin."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_count_zero_returns_zero(self):
        """When count is 0, average CPU and device times safely return 0.0."""
        ev = _DummyTimedEvent(count=0, cpu=500.0, dev=250.0)
        self.assertEqual(ev.cpu_time, 0.0)
        self.assertEqual(ev.device_time, 0.0)
        self.assertEqual(ev.cpu_time_str, "0.000us")
        self.assertEqual(ev.device_time_str, "0.000us")

    def test_nonzero_count_exact_division(self):
        """Average and formatted time properties divide totals by count accurately."""
        ev = _DummyTimedEvent(count=4, cpu=1000.0, dev=200.0, self_cpu=600.0, self_dev=120.0)
        self.assertAlmostEqual(ev.cpu_time, 250.0, places=6)
        self.assertAlmostEqual(ev.device_time, 50.0, places=6)
        self.assertEqual(ev.cpu_time_str, "250.000us")
        self.assertEqual(ev.device_time_str, "50.000us")
        self.assertEqual(ev.cpu_time_total_str, "1.000ms")
        self.assertEqual(ev.self_cpu_time_total_str, "600.000us")
        self.assertEqual(ev.self_device_time_total_str, "120.000us")

    def test_formatted_strings_across_units(self):
        """Time formatting automatically scales across us, ms, and s thresholds."""
        ev_us = _DummyTimedEvent(count=1, cpu=500.5, dev=999.0)
        self.assertEqual(ev_us.cpu_time_total_str, "500.500us")
        self.assertEqual(ev_us.device_time_total_str, "999.000us")
        ev_ms = _DummyTimedEvent(count=1, cpu=1500.0, dev=250000.0)
        self.assertEqual(ev_ms.cpu_time_total_str, "1.500ms")
        self.assertEqual(ev_ms.device_time_total_str, "250.000ms")
        ev_s = _DummyTimedEvent(count=1, cpu=2500000.0, dev=1000000.0)
        self.assertEqual(ev_s.cpu_time_total_str, "2.500s")
        self.assertEqual(ev_s.device_time_total_str, "1.000s")

    def test_cuda_time_deprecated_warning(self):
        """Accessing cuda_time returns device_time and emits a FutureWarning."""
        ev = _DummyTimedEvent(count=2, dev=300.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = ev.cuda_time
        self.assertAlmostEqual(val, ev.device_time, places=6)
        self.assertAlmostEqual(val, 150.0, places=6)
        future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
        self.assertGreaterEqual(len(future_warnings), 1)

    def test_missing_attributes_raise(self):
        """Bare FormattedTimesMixin raises AttributeError on access and TypeError on args."""
        bare = FormattedTimesMixin()
        with self.assertRaises(AttributeError):
            _ = bare.cpu_time
        with self.assertRaises(AttributeError):
            _ = bare.cpu_time_total_str
        with self.assertRaises(TypeError):
            FormattedTimesMixin(1)


if __name__ == "__main__":
    run_tests()
