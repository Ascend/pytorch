# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.get_module_to_qat_module 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.get_module_to_qat_module
API 签名：get_module_to_qat_module(backend_config) -> dict[Pattern, type[torch.nn.Module]]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空 BackendConfig 与非空 BackendConfig | 已覆盖：test_empty_config_returns_empty_dict / test_maps_pattern_to_qat_module |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | backend_config 必须为 BackendConfig，键为复杂格式 Pattern，值为模块类 | 已覆盖：test_maps_pattern_to_qat_module |
| 传参与不传参 | 该接口无默认参数，backend_config 必传 | 不适用 |
| 等价类/边界值 | 设置 qat_module、未设置 qat_module | 已覆盖：test_excludes_configs_without_qat_module |
| 正常传参场景 | 返回复杂格式 pattern 到 qat 模块类的映射 | 已覆盖：test_maps_pattern_to_qat_module / test_complex_pattern_key |
| 异常传参场景 | 非 BackendConfig 无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_module_to_qat_module 只遍历 backend_config 内部映射，非 BackendConfig 输入时
  行为由属性访问决定，无稳定契约，故不单列。
- 混合设备类型：该接口仅构造模式到模块类的映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按映射键集合与值对象同一性做断言。
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

import torch.ao.nn.qat as nnqat
import torch.nn
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig
from torch.ao.quantization.backend_config.utils import get_module_to_qat_module


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetModuleToQatModuleUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.get_module_to_qat_module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_dict(self):
        """An empty backend config yields an empty mapping."""
        self.assertEqual(get_module_to_qat_module(BackendConfig("empty")), {})

    def test_maps_pattern_to_qat_module(self):
        """A non-tuple pattern is its own key and maps to the qat module class."""
        conf = BackendPatternConfig(torch.nn.Linear).set_qat_module(nnqat.Linear)
        config = BackendConfig("qat").set_backend_pattern_config(conf)
        mapping = get_module_to_qat_module(config)
        self.assertEqual(set(mapping.keys()), {torch.nn.Linear})
        self.assertIs(mapping[torch.nn.Linear], nnqat.Linear)

    def test_complex_pattern_key(self):
        """A tuple pattern uses the reversed nested tuple format as key."""
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)).set_qat_module(nnqat.Linear)
        config = BackendConfig("complex").set_backend_pattern_config(conf)
        mapping = get_module_to_qat_module(config)
        self.assertIn((torch.nn.ReLU, torch.nn.Linear), mapping)
        self.assertIs(mapping[(torch.nn.ReLU, torch.nn.Linear)], nnqat.Linear)

    def test_excludes_configs_without_qat_module(self):
        """Configs that never set qat_module contribute nothing."""
        with_qat = BackendPatternConfig(torch.nn.Linear).set_qat_module(nnqat.Linear)
        without_qat = BackendPatternConfig(torch.nn.Conv2d).set_root_module(torch.nn.Conv2d)
        config = (
            BackendConfig("mixed")
            .set_backend_pattern_config(with_qat)
            .set_backend_pattern_config(without_qat)
        )
        mapping = get_module_to_qat_module(config)
        self.assertEqual(set(mapping.keys()), {torch.nn.Linear})


if __name__ == "__main__":
    run_tests()
