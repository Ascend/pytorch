# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.get_root_module_to_quantized_reference_module 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.get_root_module_to_quantized_reference_module
API 签名：get_root_module_to_quantized_reference_module(backend_config) -> dict[type[Module], type[Module]]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空 BackendConfig 与非空 BackendConfig | 已覆盖：test_empty_config_returns_empty_dict / test_maps_root_module_to_reference_module |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | backend_config 必须为 BackendConfig，键值均为 torch.nn.Module 子类 | 已覆盖：test_maps_root_module_to_reference_module |
| 传参与不传参 | 该接口无默认参数，backend_config 必传 | 不适用 |
| 等价类/边界值 | 同时设置、仅设置 root_module、仅设置 reference_quantized_module | 已覆盖：test_requires_both_fields |
| 正常传参场景 | 返回 root_module 到参考量化模块类的映射 | 已覆盖：test_maps_root_module_to_reference_module |
| 异常传参场景 | 非 BackendConfig 无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_root_module_to_quantized_reference_module 直接读取 backend_config.configs，
  非 BackendConfig 输入时行为由属性访问决定，无稳定契约，故不单列。
- 混合设备类型：该接口仅构造类到类的映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按映射键值对象同一性做断言。
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

import torch.ao.nn.quantized.reference as nnqr
import torch.nn
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig
from torch.ao.quantization.backend_config.utils import (
    get_root_module_to_quantized_reference_module,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetRootModuleToQuantizedReferenceModuleUt(TestCase):
    """Functional tests for get_root_module_to_quantized_reference_module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_dict(self):
        """An empty backend config yields an empty mapping."""
        self.assertEqual(get_root_module_to_quantized_reference_module(BackendConfig("empty")), {})

    def test_maps_root_module_to_reference_module(self):
        """A config setting both fields yields a root->reference mapping."""
        conf = BackendPatternConfig(torch.nn.Linear)
        conf.set_root_module(torch.nn.Linear)
        conf.set_reference_quantized_module(nnqr.Linear)
        config = BackendConfig("linear").set_backend_pattern_config(conf)
        mapping = get_root_module_to_quantized_reference_module(config)
        self.assertEqual(set(mapping.keys()), {torch.nn.Linear})
        self.assertIs(mapping[torch.nn.Linear], nnqr.Linear)

    def test_requires_both_fields(self):
        """Configs missing either root_module or reference module are skipped."""
        only_root = BackendPatternConfig(torch.nn.Conv2d).set_root_module(torch.nn.Conv2d)
        only_reference = BackendPatternConfig(torch.nn.ReLU).set_reference_quantized_module(nnqr.Linear)
        config = BackendConfig("partial")
        config.set_backend_pattern_config(only_root)
        config.set_backend_pattern_config(only_reference)
        self.assertEqual(get_root_module_to_quantized_reference_module(config), {})


if __name__ == "__main__":
    run_tests()
