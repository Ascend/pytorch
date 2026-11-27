# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.backend_config.utils.get_fuser_method_mapping 接口功能正确性
API 名称：torch.ao.quantization.backend_config.utils.get_fuser_method_mapping
API 签名：get_fuser_method_mapping(backend_config) -> dict[Pattern, nn.Sequential | Callable]

覆盖维度表：
| 覆盖维度      | 说明                               | 覆盖情况                                                    |
|---------------|------------------------------------|-------------------------------------------------------------|
| 空/非空       | 空 BackendConfig 与非空            | 已覆盖：test_empty_config_returns_empty_dict /              |
|               | BackendConfig                      | test_maps_complex_pattern_to_reversed_fuser                 |
| 枚举选项      | 该接口无枚举参数                   | 不适用                                                      |
| 参数类型      | backend_config 必须为              | 已覆盖：test_maps_complex_pattern_to_reversed_fuser         |
|               | BackendConfig，fuser_method        |                                                             |
|               | 为可调用对象                       |                                                             |
| 传参与不传参  | 该接口无默认参数，backend_config   | 不适用                                                      |
|               | 必传                               |                                                             |
| 等价类/边界值 | 2 元组模式反转、3 元组模式反转、   | 已覆盖：test_three_element_pattern_reverses_argument_order  |
|               | 未设置 fuser_method                | / test_excludes_configs_without_fuser_method                |
| 正常传参场景  | 键为反转嵌套元组格式，             | 已覆盖：test_maps_complex_pattern_to_reversed_fuser         |
|               | 值为反转后的 fuser 方法            |                                                             |
| 异常传参场景  | 非 BackendConfig 无稳定异常路径    | 未覆盖，原因见下                                            |
| 混合设备类型  | 纯 Python 结构工具，无设备执行路径 | 不适用                                                      |

未覆盖项及原因：
- 异常传参场景：get_fuser_method_mapping 内部依赖 _get_fuser_method_in_reversed_nested_tuple_format，
  仅对非法 pattern 长度抛 ValueError，该路径由 BackendPatternConfig 自身约束覆盖，故不在此单列。
- 混合设备类型：该接口仅构造模式到可调用对象的映射，不涉及 Tensor 或设备，无 CPU/NPU 对照路径。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：本接口为纯 Python 结构工具（A 类），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按映射键、可调用对象及调用结果的实参顺序做断言。
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

import torch.ao.nn.intrinsic as nni
import torch.nn
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig
from torch.ao.quantization.backend_config.utils import get_fuser_method_mapping


def _fuse_linear_relu(is_qat, linear, relu):
    return nni.LinearReLU(linear, relu)


def _fuse_three(is_qat, conv, bn, relu):
    return (conv, bn, relu)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetFuserMethodMappingUt(TestCase):
    """Functional tests for torch.ao.quantization.backend_config.utils.get_fuser_method_mapping."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_config_returns_empty_dict(self):
        """An empty backend config yields an empty mapping."""
        self.assertEqual(get_fuser_method_mapping(BackendConfig("empty")), {})

    def test_maps_complex_pattern_to_reversed_fuser(self):
        """A 2-tuple pattern maps to a fuser whose argument order was reversed."""
        conf = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)).set_fuser_method(_fuse_linear_relu)
        config = BackendConfig("two").set_backend_pattern_config(conf)
        mapping = get_fuser_method_mapping(config)
        key = (torch.nn.ReLU, torch.nn.Linear)
        self.assertIn(key, mapping)
        fuser = mapping[key]
        self.assertTrue(callable(fuser))
        self.assertIsNot(fuser, _fuse_linear_relu)
        linear = torch.nn.Linear(3, 4)
        relu = torch.nn.ReLU()
        fused = fuser(False, relu, linear)
        self.assertIsInstance(fused, nni.LinearReLU)
        self.assertIs(fused[0], linear)
        self.assertIs(fused[1], relu)

    def test_excludes_configs_without_fuser_method(self):
        """Configs without fuser_method are skipped entirely."""
        with_fused = BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU)).set_fused_module(nni.LinearReLU)
        config = BackendConfig("skip").set_backend_pattern_config(with_fused)
        mapping = get_fuser_method_mapping(config)
        self.assertEqual(mapping, {})

    def test_three_element_pattern_reverses_argument_order(self):
        """A 3-tuple pattern is reversed into (c, (b, a)) and args are reordered."""
        conf = BackendPatternConfig(
            (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU)
        ).set_fuser_method(_fuse_three)
        config = BackendConfig("three").set_backend_pattern_config(conf)
        mapping = get_fuser_method_mapping(config)
        key = (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
        self.assertIn(key, mapping)
        self.assertEqual(mapping[key](False, "relu", ("bn", "conv")), ("conv", "bn", "relu"))


if __name__ == "__main__":
    run_tests()
