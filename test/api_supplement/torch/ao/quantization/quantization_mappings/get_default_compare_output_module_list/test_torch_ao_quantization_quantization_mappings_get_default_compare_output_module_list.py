# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantization_mappings.get_default_compare_output_module_list 接口功能正确性
API 名称：torch.ao.quantization.quantization_mappings.get_default_compare_output_module_list
API 签名：get_default_compare_output_module_list() -> set[Callable]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 无入参，返回非空集合 | 已覆盖：test_returns_non_empty_set |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | 无入参；返回集合元素为模块类（type） | 已覆盖：test_every_element_is_a_class |
| 传参与不传参 | 该接口无参数 | 不适用 |
| 等价类/边界值 | 已知类命中 / 未知类不命中 | 已覆盖：test_contains_float_and_quantized_classes / test_excludes_unknown_class |
| 正常传参场景 | 返回静态/QAT/动态映射键值并集加传播白名单 | 已覆盖：test_matches_mapping_union / test_contains_propagate_include_list |
| 异常传参场景 | 该接口无入参，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_default_compare_output_module_list 不接收参数，内部仅做集合并集与深拷贝，
  不存在可由调用方构造的异常路径，故不单列。
- 混合设备类型：该接口仅构造模块类集合，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 映射工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回类型、元素类型与集合内容做断言。
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

import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.qat as nnqat
import torch.nn
from torch.ao.quantization.quantization_mappings import get_default_compare_output_module_list


def _expected_union():
    """Union of all mapping keys/values plus the qconfig propagation allowlist."""
    qm = torch.ao.quantization.quantization_mappings
    static = qm.DEFAULT_STATIC_QUANT_MODULE_MAPPINGS
    qat = qm.DEFAULT_QAT_MODULE_MAPPINGS
    dynamic = qm.DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS
    result = set()
    for mapping in (static, qat, dynamic):
        result |= set(mapping.keys())
        result |= set(mapping.values())
    result |= {torch.nn.Sequential}
    return result


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetDefaultCompareOutputModuleListUt(TestCase):
    """Functional tests for quantization_mappings.get_default_compare_output_module_list."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_non_empty_set(self):
        """The interface returns a non-empty set."""
        result = get_default_compare_output_module_list()
        self.assertIsInstance(result, set)
        self.assertGreater(len(result), 0)

    def test_every_element_is_a_class(self):
        """Every element is a module class object."""
        for item in get_default_compare_output_module_list():
            self.assertIsInstance(item, type)

    def test_contains_float_and_quantized_classes(self):
        """Both float module classes and their quantized counterparts are present."""
        result = get_default_compare_output_module_list()
        self.assertIn(torch.nn.Linear, result)
        self.assertIn(torch.nn.Conv2d, result)
        self.assertIn(nnq.Linear, result)
        self.assertIn(nnqd.Linear, result)
        self.assertIn(nnqat.Conv2d, result)
        self.assertIn(nniq.ConvReLU2d, result)

    def test_contains_propagate_include_list(self):
        """The qconfig propagation allowlist member (nn.Sequential) is included."""
        self.assertIn(torch.nn.Sequential, get_default_compare_output_module_list())

    def test_excludes_unknown_class(self):
        """A class outside every mapping is not present."""
        self.assertNotIn(torch.nn.Identity, get_default_compare_output_module_list())

    def test_matches_mapping_union(self):
        """The set equals the union of static/QAT/dynamic keys and values plus Sequential."""
        result = get_default_compare_output_module_list()
        self.assertEqual(result, _expected_union())

    def test_returns_new_object_not_shared(self):
        """Each call returns an independent object; mutation does not leak."""
        first = get_default_compare_output_module_list()
        second = get_default_compare_output_module_list()
        self.assertIsNot(first, second)
        first.add(torch.nn.Identity)
        self.assertNotIn(torch.nn.Identity, second)
        self.assertNotIn(torch.nn.Identity, get_default_compare_output_module_list())


if __name__ == "__main__":
    run_tests()
