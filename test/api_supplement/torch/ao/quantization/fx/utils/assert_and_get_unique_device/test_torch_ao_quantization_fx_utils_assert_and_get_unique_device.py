# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.assert_and_get_unique_device 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.assert_and_get_unique_device
API 签名：assert_and_get_unique_device(module) -> torch.device | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 无参数/无 buffer 的模块 vs 含 buffer 的模块 | 已覆盖：test_empty_module_returns_none / test_single_cpu_buffer_returns_cpu |
| 枚举选项 | 单一设备 / 多设备两类结果 | 已覆盖：test_single_cpu_buffer_returns_cpu / test_multiple_real_devices_raises |
| 参数类型 | 入参为 nn.Module（含 Parameters/Buffers） | 已覆盖：test_single_parameter_returns_cpu |
| 传参与不传参 | module 为唯一必填参数 | 不适用 |
| 等价类/边界值 | cpu 与 meta 混合的特殊分支 | 已覆盖：test_cpu_and_meta_warns_and_returns_cpu |
| 正常传参场景 | 单一设备模块返回该设备 | 已覆盖：test_single_cpu_buffer_returns_cpu |
| 异常传参场景 | 多设备模块抛 AssertionError | 已覆盖：test_multiple_real_devices_raises |
| 混合设备类型 | cpu 与 meta 混合的设备集合 | 已覆盖：test_cpu_and_meta_warns_and_returns_cpu |

未覆盖项及原因：
- 多设备抛错用例在无加速器的机器上会 skipTest：构造两种「真实加速设备」需要可用 NPU/CUDA，
  属环境不具备而非 API 缺陷；在有 NPU 的目标环境中会真实执行并断言 AssertionError。
- 枚举选项全覆盖：加速设备类型（npu/cuda）随机器而异，仅覆盖单一设备与冲突设备两类结果。

注意：本 API 为纯 Python 设备探测工具（类别 A），不对返回值做 CPU/NPU 数值 Golden 对照，
     而是按返回的 torch.device、警告副作用与多设备异常断言。
"""
import os
import unittest
import warnings

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

import torch.nn as nn
from torch.ao.quantization.fx.utils import assert_and_get_unique_device


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAssertAndGetUniqueDeviceUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.assert_and_get_unique_device."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_module_returns_none(self):
        self.assertIsNone(assert_and_get_unique_device(nn.Module()))

    def test_single_parameter_returns_cpu(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(3))
        self.assertEqual(assert_and_get_unique_device(module), torch.device("cpu"))

    def test_single_cpu_buffer_returns_cpu(self):
        module = nn.Module()
        module.register_buffer("buf", torch.zeros(3))
        self.assertEqual(assert_and_get_unique_device(module), torch.device("cpu"))

    def test_cpu_and_meta_warns_and_returns_cpu(self):
        module = nn.Module()
        module.register_buffer("cpu_buf", torch.zeros(3))
        module.register_buffer("meta_buf", torch.zeros(3, device="meta"))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            device = assert_and_get_unique_device(module)
        self.assertEqual(device, torch.device("cpu"))
        self.assertGreaterEqual(len(records), 1)
        self.assertIn("meta", str(records[0].message))

    def test_single_accelerator_device_returned(self):
        if self.device_name == "cpu":
            self.skipTest("No accelerator device available to construct a device module")
        module = nn.Module()
        module.register_buffer("buf", torch.zeros(3, device=self.device_name))
        device = assert_and_get_unique_device(module)
        self.assertEqual(device.type, self.device_name)

    def test_multiple_real_devices_raises(self):
        if self.device_name == "cpu":
            self.skipTest("No accelerator device available to construct a mixed-device module")
        module = nn.Module()
        module.register_buffer("dev_buf", torch.zeros(3, device=self.device_name))
        module.register_buffer("meta_buf", torch.zeros(3, device="meta"))
        with self.assertRaises(AssertionError):
            assert_and_get_unique_device(module)

    def test_cached_result_is_stable(self):
        module = nn.Module()
        module.register_buffer("buf", torch.zeros(3))
        first = assert_and_get_unique_device(module)
        second = assert_and_get_unique_device(module)
        self.assertEqual(first, second)
        self.assertEqual(second, torch.device("cpu"))


if __name__ == "__main__":
    run_tests()
