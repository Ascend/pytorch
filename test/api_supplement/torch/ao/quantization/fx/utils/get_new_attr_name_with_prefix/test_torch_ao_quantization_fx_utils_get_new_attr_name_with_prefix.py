# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.get_new_attr_name_with_prefix 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.get_new_attr_name_with_prefix
API 签名：get_new_attr_name_with_prefix(prefix) -> Callable[[Module], str]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 前缀为空字符串与普通字符串 | 已覆盖：test_empty_prefix_returns_index / test_returns_prefix_plus_index |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | 入参为 str，返回 Callable，内部入参为 nn.Module | 已覆盖：test_returns_callable |
| 传参与不传参 | prefix 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 属性名未被占用（返回 0）与已被占用（递增） | 已覆盖：test_skips_existing_attributes / test_increments_over_multiple_existing |
| 正常传参场景 | 生成未占用属性名 | 已覆盖：test_returns_prefix_plus_index |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯命名工具，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：函数仅做属性名生成，无稳定异常路径。
- 枚举选项：该 API 无枚举参数。
- 混合设备类型：纯命名工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回名称字符串与可调用性断言。
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

import torch.nn as nn
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetNewAttrNameWithPrefixUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.get_new_attr_name_with_prefix."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_callable(self):
        factory = get_new_attr_name_with_prefix("_observer")
        self.assertTrue(callable(factory))

    def test_returns_prefix_plus_index(self):
        factory = get_new_attr_name_with_prefix("_observer")
        name = factory(nn.Module())
        self.assertEqual(name, "_observer0")

    def test_empty_prefix_returns_index(self):
        factory = get_new_attr_name_with_prefix("")
        self.assertEqual(factory(nn.Module()), "0")

    def test_skips_existing_attributes(self):
        module = nn.Module()
        module._observer0 = 1
        factory = get_new_attr_name_with_prefix("_observer")
        self.assertEqual(factory(module), "_observer1")

    def test_increments_over_multiple_existing(self):
        module = nn.Module()
        module._observer0 = 1
        module._observer1 = 2
        module._observer2 = 3
        factory = get_new_attr_name_with_prefix("_observer")
        self.assertEqual(factory(module), "_observer3")

    def test_dots_replaced_with_underscores(self):
        factory = get_new_attr_name_with_prefix("a.b.c")
        self.assertEqual(factory(nn.Module()), "a_b_c0")

    def test_generated_name_is_free_on_module(self):
        module = nn.Module()
        module._observer0 = 1
        factory = get_new_attr_name_with_prefix("_observer")
        name = factory(module)
        self.assertFalse(hasattr(module, name))


if __name__ == "__main__":
    run_tests()
