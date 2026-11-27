# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.remove_boolean_dispatch_from_name 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.remove_boolean_dispatch_from_name
API 签名：remove_boolean_dispatch_from_name(p) -> Any

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 适用场景无空值语义（无参数可选空） | 不适用 |
| 枚举选项 | 全部 8 个已映射 boolean_dispatch 算子分支 | 已覆盖：test_fractional_max_pool_names / test_max_pool_names / test_adaptive_max_pool_names |
| 参数类型 | p 可为布尔分派函数、普通函数、类、字符串 | 已覆盖：test_non_dispatch_callable_unchanged / test_string_passthrough |
| 传参与不传参 | 该接口无默认参数，p 必传 | 不适用 |
| 等价类/边界值 | 8 个映射算子、非分派对象、未知分派函数 | 已覆盖：test_max_pool_names / test_unknown_boolean_dispatch_raises |
| 正常传参场景 | 映射算子返回可读名，其它对象原样返回 | 已覆盖：test_non_dispatch_callable_unchanged |
| 异常传参场景 | 未知 boolean_dispatch 函数触发 AssertionError | 已覆盖：test_unknown_boolean_dispatch_raises |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 空/非空：该接口对任意对象均有确定返回，没有「空值」这一独立语义面，故不单列。
- 混合设备类型：该接口仅做函数对象到字符串的映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回字符串内容、对象同一性与异常做断言。
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
from torch.ao.quantization.backend_config.utils import remove_boolean_dispatch_from_name


def _make_boolean_dispatch():
    def boolean_dispatch(*args, **kwargs):
        return None

    return boolean_dispatch


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRemoveBooleanDispatchFromNameUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.remove_boolean_dispatch_from_name."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_fractional_max_pool_names(self):
        """Fractional max pool dispatch functions map to dotted names."""
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.fractional_max_pool2d),
            "torch.nn.functional.fractional_max_pool2d",
        )
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.fractional_max_pool3d),
            "torch.nn.functional.fractional_max_pool3d",
        )

    def test_max_pool_names(self):
        """max_pool1d/2d/3d dispatch functions map to dotted names."""
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.max_pool1d),
            "torch.nn.functional.max_pool1d",
        )
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.max_pool2d),
            "torch.nn.functional.max_pool2d",
        )
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.max_pool3d),
            "torch.nn.functional.max_pool3d",
        )

    def test_adaptive_max_pool_names(self):
        """adaptive_max_pool1d/2d/3d dispatch functions map to dotted names."""
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.adaptive_max_pool1d),
            "torch.nn.functional.adaptive_max_pool1d",
        )
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.adaptive_max_pool2d),
            "torch.nn.functional.adaptive_max_pool2d",
        )
        self.assertEqual(
            remove_boolean_dispatch_from_name(torch.nn.functional.adaptive_max_pool3d),
            "torch.nn.functional.adaptive_max_pool3d",
        )

    def test_non_dispatch_callable_unchanged(self):
        """Non-dispatch callables and classes are returned unchanged."""
        self.assertIs(remove_boolean_dispatch_from_name(torch.add), torch.add)
        self.assertIs(remove_boolean_dispatch_from_name(torch.nn.Linear), torch.nn.Linear)

    def test_string_passthrough(self):
        """A plain string without 'boolean_dispatch' is returned unchanged."""
        self.assertEqual(remove_boolean_dispatch_from_name("torch.add"), "torch.add")

    def test_unknown_boolean_dispatch_raises(self):
        """An unmapped boolean_dispatch function raises AssertionError."""
        with self.assertRaises(AssertionError):
            remove_boolean_dispatch_from_name(_make_boolean_dispatch())


if __name__ == "__main__":
    run_tests()
