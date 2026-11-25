# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.benchmark.utils.fuzzer.prod 接口功能正确性
API 名称：torch.utils.benchmark.utils.fuzzer.prod
API 签名：torch.utils.benchmark.utils.fuzzer.prod(values, base=1)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空列表/空元组与非空可迭代对象        | 已覆盖（empty_iterable、list_and_tuple_ints）      |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | list、tuple、生成器、float 及非可迭代 | 已覆盖（list_and_tuple_ints、floats_cast_to_int、non_iterable_raises） |
| 传参与不传参     | base 默认值与显式传参、缺必选参数    | 已覆盖（list_and_tuple_ints、custom_base、missing_arg_raises） |
| 等价类/边界值    | 单元素、含 0、负数及超大整数边界     | 已覆盖（single_and_zero_and_negative、large_int_no_overflow） |
| 正常传参场景     | 常规可迭代对象连乘与自定义 base      | 已覆盖（list_and_tuple_ints、custom_base、floats_cast_to_int 等） |
| 异常传参场景     | 传入非可迭代对象与缺参触发 TypeError | 已覆盖（non_iterable_raises、missing_arg_raises）  |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 枚举选项：接口仅接收数值可迭代对象与整数 base，无枚举参数。
- 混合设备类型：纯 Python 整数连乘辅助函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.benchmark.utils.fuzzer import prod

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
class TestFuzzerProd(TestCase):
    """Test cases for torch.utils.benchmark.utils.fuzzer.prod."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_list_and_tuple_ints(self):
        """Computes product of int list, tuple, and generator correctly."""
        self.assertEqual(prod([2, 3, 4]), 24)
        self.assertEqual(prod((5, 6)), 30)
        self.assertEqual(prod(x for x in [2, 5, 7]), 70)

    def test_empty_iterable(self):
        """Empty iterable returns base (default 1 or explicit base)."""
        self.assertEqual(prod([]), 1)
        self.assertEqual(prod(()), 1)
        self.assertEqual(prod([], base=7), 7)

    def test_custom_base(self):
        """Custom base multiplies into the product."""
        self.assertEqual(prod([2, 3], base=10), 60)
        self.assertEqual(prod((4,), base=0), 0)

    def test_single_and_zero_and_negative(self):
        """Single element, zero element, and negative ints behave accurately."""
        self.assertEqual(prod([9]), 9)
        self.assertEqual(prod([3, 0, 5]), 0)
        self.assertEqual(prod([-2, 3, -4]), 24)

    def test_floats_cast_to_int(self):
        """Float elements are truncated via int() during reduction."""
        res = prod([2.5, 4.0])
        self.assertIsInstance(res, int)
        self.assertEqual(res, 8)

    def test_large_int_no_overflow(self):
        """Large integers beyond int64 max do not overflow in Python int."""
        big = 2 ** 40
        self.assertEqual(prod([big, big]), 2 ** 80)

    def test_non_iterable_raises(self):
        """Passing a non-iterable value raises TypeError."""
        with self.assertRaises(TypeError):
            prod(123)

    def test_missing_arg_raises(self):
        """Calling prod with no arguments raises TypeError."""
        with self.assertRaises(TypeError):
            prod()


if __name__ == "__main__":
    run_tests()
