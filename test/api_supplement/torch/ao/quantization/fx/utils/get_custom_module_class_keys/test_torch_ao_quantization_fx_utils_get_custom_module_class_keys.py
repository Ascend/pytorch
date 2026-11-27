# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.get_custom_module_class_keys 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.get_custom_module_class_keys
API 签名：get_custom_module_class_keys(custom_module_mapping) -> list

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空映射与含多个 QuantType 的映射 | 已覆盖：test_empty_mapping_returns_empty_list / test_collects_keys_across_quant_types |
| 枚举选项 | STATIC / DYNAMIC / WEIGHT_ONLY 三种 QuantType | 已覆盖：test_collects_keys_across_quant_types |
| 参数类型 | 入参为 dict[QuantType, dict[type, type]] | 已覆盖：test_returns_list_type |
| 传参与不传参 | custom_module_mapping 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 同一类在多个 QuantType 下重复出现 | 已覆盖：test_dedups_shared_keys |
| 正常传参场景 | 提取并去重所有内部键 | 已覆盖：test_collects_keys_across_quant_types |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯字典遍历，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：入参为映射时函数总能返回列表；非法类型会在 `.keys()` 处抛 AttributeError，
  属底层而非本 API 稳定契约，故不单列。
- 混合设备类型：纯 Python 字典工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回集合内容与类型断言。
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

from torch.ao.quantization import QuantType
from torch.ao.quantization.fx.utils import get_custom_module_class_keys


class _StaticCustom:
    pass


class _DynamicCustom:
    pass


class _WeightOnlyCustom:
    pass


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetCustomModuleClassKeysUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.get_custom_module_class_keys."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_mapping_returns_empty_list(self):
        self.assertEqual(get_custom_module_class_keys({}), [])

    def test_returns_list_type(self):
        mapping = {QuantType.STATIC: {_StaticCustom: object}}
        result = get_custom_module_class_keys(mapping)
        self.assertIsInstance(result, list)

    def test_collects_keys_across_quant_types(self):
        mapping = {
            QuantType.STATIC: {_StaticCustom: object},
            QuantType.DYNAMIC: {_DynamicCustom: object},
            QuantType.WEIGHT_ONLY: {_WeightOnlyCustom: object},
        }
        result = get_custom_module_class_keys(mapping)
        self.assertEqual(
            set(result), {_StaticCustom, _DynamicCustom, _WeightOnlyCustom}
        )

    def test_dedups_shared_keys(self):
        mapping = {
            QuantType.STATIC: {_StaticCustom: object},
            QuantType.WEIGHT_ONLY: {_StaticCustom: object},
        }
        result = get_custom_module_class_keys(mapping)
        self.assertEqual(result.count(_StaticCustom), 1)
        self.assertEqual(len(result), 1)

    def test_missing_quant_types_ignored(self):
        mapping = {QuantType.STATIC: {_StaticCustom: object}}
        result = get_custom_module_class_keys(mapping)
        self.assertEqual(result, [_StaticCustom])


if __name__ == "__main__":
    run_tests()
