# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantization_mappings.no_observer_set 接口功能正确性
API 名称：torch.ao.quantization.quantization_mappings.no_observer_set
API 签名：no_observer_set() -> set[Any]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 无入参，返回非空集合 | 已覆盖：test_returns_non_empty_set |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | 无入参；返回集合元素为不可插 observer 的模块类 | 已覆盖：test_every_element_is_a_class |
| 传参与不传参 | 该接口无参数 | 不适用 |
| 等价类/边界值 | 集合恰好为 LSTM 与 MultiheadAttention 两类 | 已覆盖：test_matches_expected_members |
| 正常传参场景 | 返回 quantizable LSTM / MultiheadAttention 集合 | 已覆盖：test_contains_quantizable_modules |
| 异常传参场景 | 该接口无入参，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 集合工厂，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：no_observer_set 不接收参数，内部仅构造一个固定集合，无调用方可构造的异常路径，故不单列。
- 混合设备类型：该接口仅返回模块类集合，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 集合工厂（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回类型与集合元素做断言。
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

import torch.ao.nn.quantizable as quantizable
import torch.nn
from torch.ao.quantization.quantization_mappings import no_observer_set

_EXPECTED_MEMBERS = frozenset({quantizable.LSTM, quantizable.MultiheadAttention})


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestNoObserverSetUt(TestCase):
    """Functional tests for quantization_mappings.no_observer_set."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_non_empty_set(self):
        """The interface returns a non-empty set."""
        result = no_observer_set()
        self.assertIsInstance(result, set)
        self.assertGreater(len(result), 0)

    def test_every_element_is_a_class(self):
        """Every element is a module class object."""
        for item in no_observer_set():
            self.assertIsInstance(item, type)

    def test_contains_quantizable_modules(self):
        """The quantizable LSTM and MultiheadAttention classes are present."""
        result = no_observer_set()
        self.assertIn(quantizable.LSTM, result)
        self.assertIn(quantizable.MultiheadAttention, result)

    def test_matches_expected_members(self):
        """The set contains exactly the documented module classes."""
        result = no_observer_set()
        self.assertEqual(result, set(_EXPECTED_MEMBERS))
        self.assertEqual(len(result), 2)

    def test_excludes_regular_module(self):
        """An unrelated module class is not part of the no-observer set."""
        self.assertNotIn(torch.nn.Linear, no_observer_set())

    def test_returns_new_object_not_shared(self):
        """Each call returns an independent set; mutation does not leak."""
        first = no_observer_set()
        second = no_observer_set()
        self.assertIsNot(first, second)
        first.clear()
        self.assertEqual(len(second), 2)
        self.assertEqual(len(no_observer_set()), 2)


if __name__ == "__main__":
    run_tests()
