# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.pattern_to_human_readable 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.pattern_to_human_readable
API 签名：pattern_to_human_readable(p) -> Any

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空元组 () 与非空模式 | 已覆盖：test_tuple_recursion / test_nested_tuple_recursion |
| 枚举选项 | 元组/字符串/可调用对象/其他类型分支 | 已覆盖：test_string_passthrough / test_callable_non_dispatch / test_other_type_passthrough |
| 参数类型 | p 可为 tuple、str、函数、其他对象 | 已覆盖：test_string_passthrough / test_other_type_passthrough |
| 传参与不传参 | 该接口无默认参数，p 必传 | 不适用 |
| 等价类/边界值 | 单层元组、嵌套元组、已可读字符串、boolean_dispatch 函数 | 已覆盖：test_nested_tuple_recursion / test_boolean_dispatch_op_name |
| 正常传参场景 | 元组递归转换、字符串原样返回、非 dispatch 可调用原样返回 | 已覆盖：test_tuple_recursion / test_callable_non_dispatch |
| 异常传参场景 | 未知 boolean_dispatch 函数触发 AssertionError | 已覆盖：test_unknown_boolean_dispatch_raises |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 混合设备类型：该接口仅做模式对象到可读表示的转换，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回对象内容、同一性与异常做断言。
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

import torch.nn
import torch.nn.functional
from torch.ao.quantization.backend_config.utils import pattern_to_human_readable


def _make_boolean_dispatch():
    def boolean_dispatch(*args, **kwargs):
        return None

    return boolean_dispatch


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestPatternToHumanReadableUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.pattern_to_human_readable."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_tuple_recursion(self):
        """A tuple pattern is converted element-wise and stays a tuple."""
        result = pattern_to_human_readable((torch.nn.Linear, torch.nn.ReLU))
        self.assertEqual(result, (torch.nn.Linear, torch.nn.ReLU))
        self.assertIsInstance(result, tuple)

    def test_nested_tuple_recursion(self):
        """Nested tuples are recursed through, preserving structure."""
        result = pattern_to_human_readable((torch.add, (torch.nn.Linear, torch.nn.ReLU)))
        self.assertEqual(result, (torch.add, (torch.nn.Linear, torch.nn.ReLU)))

    def test_string_passthrough(self):
        """String method names are already human readable and returned as is."""
        self.assertEqual(pattern_to_human_readable("torch.add"), "torch.add")

    def test_callable_non_dispatch(self):
        """A non-dispatch callable is returned unchanged."""
        self.assertIs(pattern_to_human_readable(torch.add), torch.add)
        self.assertIs(pattern_to_human_readable(torch.nn.Linear), torch.nn.Linear)

    def test_boolean_dispatch_op_name(self):
        """A known boolean_dispatch op becomes its dotted human readable name."""
        result = pattern_to_human_readable(torch.nn.functional.max_pool2d)
        self.assertEqual(result, "torch.nn.functional.max_pool2d")

    def test_other_type_passthrough(self):
        """A value that is neither tuple, str nor dispatch function is unchanged."""
        self.assertEqual(pattern_to_human_readable(3), 3)

    def test_unknown_boolean_dispatch_raises(self):
        """An unmapped boolean_dispatch function raises AssertionError."""
        with self.assertRaises(AssertionError):
            pattern_to_human_readable(_make_boolean_dispatch())


if __name__ == "__main__":
    run_tests()
