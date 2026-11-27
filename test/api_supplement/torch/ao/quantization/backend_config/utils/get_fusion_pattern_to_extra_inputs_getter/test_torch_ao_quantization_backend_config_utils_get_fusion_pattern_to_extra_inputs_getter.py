# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.get_fusion_pattern_to_extra_inputs_getter 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.get_fusion_pattern_to_extra_inputs_getter
API 签名：get_fusion_pattern_to_extra_inputs_getter(backend_config) -> dict[Pattern, Callable]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空 BackendConfig 与非空 BackendConfig | 已覆盖：test_empty_config_returns_empty_dict / test_maps_pattern_to_extra_inputs_getter |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 参数类型 | backend_config 必须为 BackendConfig，getter 为可调用对象 | 已覆盖：test_maps_pattern_to_extra_inputs_getter |
| 传参与不传参 | 该接口无默认参数，backend_config 必传 | 不适用 |
| 等价类/边界值 | 设置的 getter、未设置的 getter | 已覆盖：test_excludes_configs_without_extra_inputs_getter |
| 正常传参场景 | 返回模式到 extra_inputs_getter 的映射且可调用 | 已覆盖：test_maps_pattern_to_extra_inputs_getter |
| 异常传参场景 | 非 BackendConfig 无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_fusion_pattern_to_extra_inputs_getter 只读取内部属性 _extra_inputs_getter，
  非 BackendConfig 输入时行为由属性访问决定，无稳定契约，故不单列。
- 混合设备类型：该接口仅构造模式到可调用对象的映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按映射键、可调用对象同一性及其调用结果做断言。
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
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_extra_inputs_getter


def _extra_inputs_getter(pattern):
    return ["extra_input"]


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetFusionPatternToExtraInputsGetterUt(TestCase):
    """Functional tests for get_fusion_pattern_to_extra_inputs_getter."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_dict(self):
        """An empty backend config yields an empty mapping."""
        self.assertEqual(get_fusion_pattern_to_extra_inputs_getter(BackendConfig("empty")), {})

    def test_maps_pattern_to_extra_inputs_getter(self):
        """A configured extra_inputs_getter is returned under its pattern key."""
        conf = BackendPatternConfig(torch.nn.Linear)._set_extra_inputs_getter(_extra_inputs_getter)
        config = BackendConfig("extra").set_backend_pattern_config(conf)
        mapping = get_fusion_pattern_to_extra_inputs_getter(config)
        self.assertIn(torch.nn.Linear, mapping)
        self.assertIs(mapping[torch.nn.Linear], _extra_inputs_getter)
        self.assertEqual(mapping[torch.nn.Linear](("add", "input", "conv")), ["extra_input"])

    def test_excludes_configs_without_extra_inputs_getter(self):
        """Configs without an extra_inputs_getter are skipped."""
        conf = BackendPatternConfig(torch.add)
        config = BackendConfig("none").set_backend_pattern_config(conf)
        self.assertEqual(get_fusion_pattern_to_extra_inputs_getter(config), {})


if __name__ == "__main__":
    run_tests()
