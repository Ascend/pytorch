# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.get_qat_module_classes 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.get_qat_module_classes
API 签名：get_qat_module_classes(backend_config) -> tuple[type, ...]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空 BackendConfig 与非空 BackendConfig | 已覆盖：test_empty_config_returns_empty_tuple / test_returns_qat_module_classes |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | backend_config 必须为 BackendConfig，内部 configs 提供 qat_module | 已覆盖：test_excludes_configs_without_qat_module |
| 传参与不传参 | 该接口无默认参数，backend_config 必传 | 不适用 |
| 等价类/边界值 | 无 qat_module 的配置、重复 qat_module | 已覆盖：test_excludes_configs_without_qat_module / test_dedupes_repeated_qat_module |
| 正常传参场景 | 返回全部去重 qat_module 类元组 | 已覆盖：test_returns_qat_module_classes / test_result_holds_exact_class_objects |
| 异常传参场景 | 非 BackendConfig 无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_qat_module_classes 直接读取 backend_config.configs，传入非 BackendConfig 对象时
  属性缺失行为由属性访问机制决定，无稳定契约，故不单列。
- 混合设备类型：该接口仅遍历配置对象并读取类属性，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回结构与类对象同一性做断言。
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

import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.nn
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig
from torch.ao.quantization.backend_config.utils import get_qat_module_classes


def _make_config():
    linear = BackendPatternConfig(torch.nn.Linear).set_qat_module(nnqat.Linear)
    conv_relu = BackendPatternConfig(
        (torch.nn.Conv2d, torch.nn.ReLU)
    ).set_qat_module(nniqat.ConvReLU2d)
    plain = BackendPatternConfig(torch.nn.Conv2d)
    return (
        BackendConfig("test")
        .set_backend_pattern_config(linear)
        .set_backend_pattern_config(conv_relu)
        .set_backend_pattern_config(plain)
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetQatModuleClassesUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.get_qat_module_classes."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_tuple(self):
        """An empty backend config yields an empty tuple."""
        result = get_qat_module_classes(BackendConfig("empty"))
        self.assertEqual(result, ())
        self.assertIsInstance(result, tuple)

    def test_returns_qat_module_classes(self):
        """All qat_module entries are collected into a tuple."""
        result = get_qat_module_classes(_make_config())
        self.assertIsInstance(result, tuple)
        self.assertCountEqual(result, [nnqat.Linear, nniqat.ConvReLU2d])
        self.assertEqual(len(result), 2)

    def test_excludes_configs_without_qat_module(self):
        """Configs that never set qat_module contribute nothing."""
        result = get_qat_module_classes(_make_config())
        self.assertNotIn(torch.nn.Linear, result)
        self.assertNotIn(torch.nn.Conv2d, result)

    def test_dedupes_repeated_qat_module(self):
        """The same qat_module registered twice appears once."""
        first = BackendPatternConfig(torch.nn.Linear).set_qat_module(nnqat.Linear)
        second = BackendPatternConfig(torch.nn.Conv2d).set_qat_module(nnqat.Linear)
        config = BackendConfig("dedup").set_backend_pattern_config(first).set_backend_pattern_config(second)
        result = get_qat_module_classes(config)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], nnqat.Linear)

    def test_result_holds_exact_class_objects(self):
        """Entries are the registered class objects, not copies or wrappers."""
        result = get_qat_module_classes(_make_config())
        for item in result:
            self.assertIsInstance(item, type)
        self.assertIn(nnqat.Linear, result)
        self.assertIn(nniqat.ConvReLU2d, result)


if __name__ == "__main__":
    run_tests()
