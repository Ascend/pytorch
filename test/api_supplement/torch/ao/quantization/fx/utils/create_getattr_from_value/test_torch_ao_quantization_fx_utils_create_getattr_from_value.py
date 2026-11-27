# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.create_getattr_from_value 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.create_getattr_from_value
API 签名：create_getattr_from_value(module, graph, prefix, value, device=None) -> Node

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | value 为标量与 Tensor 两种非空输入 | 已覆盖：test_scalar_value_stored / test_tensor_value_is_cloned |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | value 为 int / Tensor，device 为 None / torch.device | 已覆盖：test_explicit_device_used |
| 传参与不传参 | device 省略（默认 None）与显式传入 | 已覆盖：test_default_device_path / test_explicit_device_used |
| 等价类/边界值 | 同一前缀多次调用产生唯一属性名 | 已覆盖：test_repeated_calls_produce_unique_names |
| 正常传参场景 | 注册 buffer 并返回 get_attr 节点 | 已覆盖：test_returns_get_attr_node / test_registers_buffer_with_prefix |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 无多 Tensor 设备输入路径 | 不适用 |

未覆盖项及原因：
- 异常传参场景：value 为任意可转 Tensor 的对象即可成功，非法输入由 torch.tensor 抛错，
  属底层而非本 API 稳定契约，故不单列。
- 枚举选项：该 API 无枚举参数。
- 混合设备类型：单值注册工具，不涉及多 Tensor 异构设备输入。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回节点结构、buffer 内容与对象同一性断言。
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

import torch.fx
import torch.nn as nn
from torch.ao.quantization.fx.utils import create_getattr_from_value


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCreateGetattrFromValueUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.create_getattr_from_value."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_get_attr_node(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        node = create_getattr_from_value(module, graph, "_val", 3)
        self.assertEqual(node.op, "get_attr")
        self.assertIsInstance(node, torch.fx.Node)

    def test_registers_buffer_with_prefix(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        node = create_getattr_from_value(module, graph, "_my_buf", 3)
        self.assertTrue(hasattr(module, node.target))
        self.assertTrue(node.target.startswith("_my_buf"))
        self.assertEqual(node.target, "_my_buf0")

    def test_scalar_value_stored(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        node = create_getattr_from_value(module, graph, "_val", 7)
        buffer = getattr(module, node.target)
        self.assertIsInstance(buffer, torch.Tensor)
        self.assertEqual(int(buffer), 7)

    def test_tensor_value_is_cloned(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        value = torch.arange(4.0)
        node = create_getattr_from_value(module, graph, "_buf", value)
        buffer = getattr(module, node.target)
        self.assertTrue(torch.equal(buffer, value))
        # The stored buffer must be a clone, not the caller's object.
        self.assertIsNot(buffer, value)

    def test_tensor_value_is_detached(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        value = torch.arange(4.0, requires_grad=True)
        node = create_getattr_from_value(module, graph, "_buf", value)
        buffer = getattr(module, node.target)
        self.assertFalse(buffer.requires_grad)

    def test_repeated_calls_produce_unique_names(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        first = create_getattr_from_value(module, graph, "_val", 1)
        second = create_getattr_from_value(module, graph, "_val", 2)
        self.assertNotEqual(first.target, second.target)
        self.assertEqual(int(getattr(module, first.target)), 1)
        self.assertEqual(int(getattr(module, second.target)), 2)

    def test_explicit_device_used(self):
        module = nn.Module()
        graph = torch.fx.Graph()
        node = create_getattr_from_value(module, graph, "_val", 3, device=torch.device("cpu"))
        buffer = getattr(module, node.target)
        self.assertEqual(buffer.device, torch.device("cpu"))

    def test_default_device_path(self):
        module = nn.Module()
        module.register_buffer("anchor", torch.zeros(1))
        graph = torch.fx.Graph()
        node = create_getattr_from_value(module, graph, "_val", 3)
        buffer = getattr(module, node.target)
        self.assertEqual(buffer.device, torch.device("cpu"))


if __name__ == "__main__":
    run_tests()
