# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.benchmark.utils.common.trim_sigfig 接口功能正确性
API 名称：torch.utils.benchmark.utils.common.trim_sigfig
API 签名：torch.utils.benchmark.utils.common.trim_sigfig(x, n)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空容器输入                     | 未覆盖（见下）                                     |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | 浮点 x、整型/整数浮点 n 及非整型 n   | 已覆盖（positive_numbers、float_integer_n、non_int_n_raises） |
| 传参与不传参     | 两个必选参数传参与缺参               | 已覆盖（positive_numbers、missing_arg_raises）     |
| 等价类/边界值    | 正/负数、极小/大数量级与 x=0.0 边界  | 已覆盖（negative_numbers、small_and_large_magnitudes、zero_raises） |
| 正常传参场景     | 不同有效数字位数裁剪正负数           | 已覆盖（positive_numbers、negative_numbers、float_integer_n 等） |
| 异常传参场景     | 非整型 n、x=0.0 及缺参触发异常       | 已覆盖（non_int_n_raises、zero_raises、missing_arg_raises） |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 空/非空：接口仅接收标量数值与有效数字位数，无容器输入。
- 枚举选项：接口无枚举类型参数。
- 混合设备类型：纯 Python 数值处理函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.benchmark.utils.common import trim_sigfig

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
class TestTrimSigfig(TestCase):
    """Test cases for torch.utils.benchmark.utils.common.trim_sigfig."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_positive_numbers(self):
        """Positive floats are trimmed to 1, 2, or 3 significant figures."""
        res1 = trim_sigfig(3.14159, 1)
        self.assertIsInstance(res1, float)
        self.assertAlmostEqual(res1, 3.0, places=5)
        self.assertAlmostEqual(trim_sigfig(3.14159, 2), 3.1, places=5)
        self.assertAlmostEqual(trim_sigfig(3.14159, 3), 3.14, places=5)

    def test_negative_numbers(self):
        """Negative values preserve sign and trim significant figures."""
        self.assertAlmostEqual(trim_sigfig(-1234.5, 2), -1200.0, places=3)
        self.assertAlmostEqual(trim_sigfig(-0.05678, 2), -0.057, places=5)

    def test_small_and_large_magnitudes(self):
        """Both very small and large magnitudes are rounded accurately."""
        self.assertAlmostEqual(trim_sigfig(0.01234, 2), 0.012, places=6)
        self.assertAlmostEqual(trim_sigfig(98765.0, 3), 98800.0, places=1)

    def test_float_integer_n(self):
        """A float representing an exact integer (e.g. 2.0) is accepted."""
        self.assertAlmostEqual(trim_sigfig(3.14159, 2.0), 3.1, places=5)

    def test_non_int_n_raises(self):
        """Non-integer n raises AssertionError."""
        with self.assertRaises(AssertionError):
            trim_sigfig(3.14159, 1.5)

    def test_zero_raises(self):
        """Passing x=0.0 triggers log10(0) = -inf and raises OverflowError."""
        with self.assertRaises(OverflowError):
            trim_sigfig(0.0, 2)

    def test_missing_arg_raises(self):
        """Calling trim_sigfig with missing required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            trim_sigfig(3.14159)


if __name__ == "__main__":
    run_tests()
