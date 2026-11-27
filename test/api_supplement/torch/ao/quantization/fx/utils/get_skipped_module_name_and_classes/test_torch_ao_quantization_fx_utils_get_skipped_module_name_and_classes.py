# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.get_skipped_module_name_and_classes 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.get_skipped_module_name_and_classes
API 签名：get_skipped_module_name_and_classes(prepare_custom_config, is_standalone_module)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空配置与含独立/自定义模块配置 | 已覆盖：test_not_standalone_includes_standalone_and_custom |
| 枚举选项 | is_standalone_module 为 True/False 两分支 | 已覆盖：test_standalone_returns_only_non_traceable |
| 参数类型 | 入参为 PrepareCustomConfig + bool | 已覆盖：test_returns_tuple_of_two_lists |
| 传参与不传参 | 两参数均为必填 | 不适用 |
| 等价类/边界值 | 返回的是配置列表的副本而非引用 | 已覆盖：test_returns_copies_not_original_lists |
| 正常传参场景 | 顶层模块聚合独立模块与自定义模块键 | 已覆盖：test_not_standalone_includes_standalone_and_custom |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯配置解析，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：函数仅读取配置对象字段并返回列表，无稳定异常路径。
- 枚举选项：bool 两分支均已覆盖。
- 混合设备类型：纯配置解析工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 配置工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回列表的内容、拷贝语义与类型断言。
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

from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from torch.ao.quantization.fx.utils import get_skipped_module_name_and_classes


class _NonTraceableA:
    pass


class _StandaloneB:
    pass


class _FloatCustom:
    pass


class _ObservedCustom:
    pass


def _build_config():
    config = PrepareCustomConfig()
    config.set_non_traceable_module_names(["module1"])
    config.set_non_traceable_module_classes([_NonTraceableA])
    config.set_standalone_module_name("standalone_name", None, (torch.randn(2),), None, None)
    config.set_standalone_module_class(_StandaloneB, None, (torch.randn(2),), None, None)
    config.set_float_to_observed_mapping(_FloatCustom, _ObservedCustom)
    return config


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetSkippedModuleNameAndClassesUt(TestCase):
    """Functional tests for get_skipped_module_name_and_classes."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_lists(self):
        empty = PrepareCustomConfig()
        names, classes = get_skipped_module_name_and_classes(empty, False)
        self.assertEqual(names, [])
        self.assertEqual(classes, [])

    def test_returns_tuple_of_two_lists(self):
        names, classes = get_skipped_module_name_and_classes(PrepareCustomConfig(), True)
        self.assertIsInstance(names, list)
        self.assertIsInstance(classes, list)

    def test_not_standalone_includes_standalone_and_custom(self):
        config = _build_config()
        names, classes = get_skipped_module_name_and_classes(config, False)
        self.assertIn("module1", names)
        self.assertIn("standalone_name", names)
        self.assertIn(_NonTraceableA, classes)
        self.assertIn(_StandaloneB, classes)
        # Float classes of the float-to-observed mapping are added for the
        # top-level module (the observed class itself is not included).
        self.assertIn(_FloatCustom, classes)
        self.assertNotIn(_ObservedCustom, classes)

    def test_standalone_returns_only_non_traceable(self):
        config = _build_config()
        names, classes = get_skipped_module_name_and_classes(config, True)
        self.assertEqual(names, ["module1"])
        self.assertEqual(classes, [_NonTraceableA])

    def test_returns_copies_not_original_lists(self):
        config = _build_config()
        names, _ = get_skipped_module_name_and_classes(config, True)
        self.assertIsNot(names, config.non_traceable_module_names)
        names.append("mutated")
        self.assertNotIn("mutated", config.non_traceable_module_names)


if __name__ == "__main__":
    run_tests()
