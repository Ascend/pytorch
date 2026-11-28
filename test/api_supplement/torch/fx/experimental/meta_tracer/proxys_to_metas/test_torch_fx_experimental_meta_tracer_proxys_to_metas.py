# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.meta_tracer.proxys_to_metas 接口功能正确性
API 名称：torch.fx.experimental.meta_tracer.proxys_to_metas
API 签名：proxys_to_metas(v) -> Any

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的零尺寸或标量输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：meta Tensor、对应 nn.Module 或真实 FX Proxy |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：普通对象、普通 Proxy 和无元数据 MetaProxy |
| 正常传参场景 | 已覆盖：真实 MetaProxy 的已安装元数据及 device 属性 |
| 异常传参场景 | 已覆盖：非 MetaProxy 与无关联元数据的 AssertionError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 该接口用于 meta 分析；返回原输入的 override 仅验证元数据契约，不声明对应算子的数值正确性。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

from torch.fx.experimental.meta_tracer import proxys_to_metas
from torch.fx.proxy import GraphAppendingTracer
from torch.fx.experimental.meta_tracer import MetaProxy


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_proxys_to_metas_meta_proxy_and_device_attribute_map_to_meta_values(self):
        graph = torch.fx.Graph()
        proxy = MetaProxy(graph.placeholder("value"), GraphAppendingTracer(graph))
        value = torch.empty(2, 3, device="meta")
        proxy.install_tensor_meta(value)
        self.assertIs(proxys_to_metas(proxy), value)
        self.assertEqual(proxys_to_metas(proxy.device), "meta")

    def test_proxys_to_metas_plain_values_pass_through_and_unsupported_proxy_is_rejected(self):
        ordinary = [1, 2]
        self.assertIs(proxys_to_metas(ordinary), ordinary)
        self.assertIsNone(proxys_to_metas(None))
        graph = torch.fx.Graph()
        node = graph.placeholder("value")
        with self.assertRaisesRegex(AssertionError, "Expected MetaProxy"):
            proxys_to_metas(torch.fx.Proxy(node, GraphAppendingTracer(graph)))
        with self.assertRaisesRegex(AssertionError, "associated meta"):
            proxys_to_metas(MetaProxy(node, GraphAppendingTracer(graph)))


if __name__ == "__main__":
    run_tests()
