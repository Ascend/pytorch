# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.get_pattern_to_dtype_configs 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.get_pattern_to_dtype_configs
API 签名：get_pattern_to_dtype_configs(backend_config) -> dict[Pattern, list[DTypeConfig]]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 空 BackendConfig 与非空 BackendConfig | 已覆盖：test_empty_config_returns_empty_dict / test_maps_pattern_to_dtype_configs |
| 枚举选项 | dtype_configs 可为空列表或含多个 DTypeConfig | 已覆盖：test_config_without_dtype_configs_maps_to_empty_list |
| 参数类型 | backend_config 必须为 BackendConfig，值为 DTypeConfig 列表 | 已覆盖：test_maps_pattern_to_dtype_configs |
| 传参与不传参 | 该接口无默认参数，backend_config 必传 | 不适用 |
| 等价类/边界值 | 单元素/多元素 dtype_configs、元组模式复杂键 | 已覆盖：test_maps_pattern_to_dtype_configs / test_uses_complex_pattern_key |
| 正常传参场景 | 返回复杂格式 pattern 到 DTypeConfig 列表的映射 | 已覆盖：test_uses_complex_pattern_key |
| 异常传参场景 | 非 BackendConfig 无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 结构工具，无设备执行路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：get_pattern_to_dtype_configs 只遍历后端配置内部映射，非 BackendConfig 输入时
  行为由属性访问决定，无稳定契约，故不单列。
- 混合设备类型：该接口仅构造模式到 DTypeConfig 列表的映射，不涉及 Tensor 执行或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按映射键集合与列表对象同一性做断言。
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
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig, DTypeConfig
from torch.ao.quantization.backend_config.utils import get_pattern_to_dtype_configs

_DTYPE_CONFIG1 = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)
_DTYPE_CONFIG2 = DTypeConfig(input_dtype=torch.float16, output_dtype=torch.float, is_dynamic=True)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetPatternToDtypeConfigsUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.get_pattern_to_dtype_configs."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_dict(self):
        """An empty backend config yields an empty mapping."""
        self.assertEqual(get_pattern_to_dtype_configs(BackendConfig("empty")), {})

    def test_maps_pattern_to_dtype_configs(self):
        """The mapping value is the very list stored on the pattern config."""
        conf = BackendPatternConfig(torch.nn.Linear)
        conf.add_dtype_config(_DTYPE_CONFIG1)
        conf.add_dtype_config(_DTYPE_CONFIG2)
        config = BackendConfig("linear").set_backend_pattern_config(conf)
        mapping = get_pattern_to_dtype_configs(config)
        self.assertEqual(set(mapping.keys()), {torch.nn.Linear})
        self.assertIs(mapping[torch.nn.Linear], conf.dtype_configs)
        self.assertEqual(mapping[torch.nn.Linear], [_DTYPE_CONFIG1, _DTYPE_CONFIG2])

    def test_uses_complex_pattern_key(self):
        """A 2-tuple pattern is keyed by its reversed nested tuple format."""
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)).add_dtype_config(_DTYPE_CONFIG1)
        config = BackendConfig("tuple").set_backend_pattern_config(conf)
        mapping = get_pattern_to_dtype_configs(config)
        self.assertIn((torch.nn.ReLU, torch.nn.Linear), mapping)
        self.assertEqual(mapping[(torch.nn.ReLU, torch.nn.Linear)], [_DTYPE_CONFIG1])

    def test_config_without_dtype_configs_maps_to_empty_list(self):
        """A config with no dtype configs still appears, mapped to an empty list."""
        conf = BackendPatternConfig(torch.nn.Conv2d)
        config = BackendConfig("none").set_backend_pattern_config(conf)
        mapping = get_pattern_to_dtype_configs(config)
        self.assertIn(torch.nn.Conv2d, mapping)
        self.assertEqual(mapping[torch.nn.Conv2d], [])


if __name__ == "__main__":
    run_tests()
