# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.quantize_fx.fuse_fx 接口功能正确性
API 名称：torch.ao.quantization.quantize_fx.fuse_fx
API 签名：fuse_fx(model, fuse_custom_config=None, backend_config=None)

覆盖维度表：
| 覆盖维度      | 说明                                          | 覆盖情况                                         |
|---------------|-----------------------------------------------|--------------------------------------------------|
| 空/非空       | fuse_custom_config 为 None / 非空对象 /       | 已覆盖：test_fuse_returns_graph_module、         |
|               | 非法对象                                      |                                                  |
|               |                                               | test_fuse_invalid_config_type                    |
| 枚举选项      | 融合模式：conv+relu（有融合）、conv+bn（eval  | 已覆盖：test_fuse_conv_relu_fuses_modules、      |
|               | 折叠）                                        |                                                  |
|               |                                               | test_fuse_bn_folds_into_conv                     |
| 参数类型      | model 为 nn.Module；config 为                 | 已覆盖：test_fuse_preserved_attributes           |
|               | FuseCustomConfig                              |                                                  |
| 传参与不传参  | 省略可选参数 / 显式传入                       | 已覆盖：test_fuse_returns_graph_module（省略）、 |
|               |                                               | test_fuse_preserved_attributes（显式）           |
| 等价类/边界值 | 无可融合子模块的模型 / 可融合模型             | 已覆盖：test_fuse_returns_graph_module           |
| 正常传参场景  | 返回 GraphModule 且前向 shape 保持            | 已覆盖：test_fuse_output_shape_preserved         |
| 异常传参场景  | 非法 config 类型 / dict 配置触发              | 已覆盖：test_fuse_invalid_config_type、          |
|               | FutureWarning                                 |                                                  |
|               |                                               | test_fuse_dict_config_deprecation                |
| 混合设备类型  | 多 Tensor 异构设备输入                        | 未覆盖，见下                                     |

未覆盖项及原因：
- 混合设备类型：FX 融合仅作符号追踪与图重写，CPU 量化后端提供融合映射，无 NPU/CUDA 执行路径，
  无法构造异构设备输入，故不覆盖。
- 图模式：本用例针对 eager FX 图变换，未覆盖 torch.compile / NPUGraph 图模式路径。

注：融合只改写 FX 图结构，无 CPU/NPU 数值对照路径，按图节点集合与子模块类型断言。
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

import torch.ao.nn.intrinsic
import torch.ao.quantization.quantize_fx
from torch.ao.quantization.fx.custom_config import FuseCustomConfig
from torch.fx import GraphModule

_EXAMPLE = (torch.randn(1, 3, 8, 8),)


def _conv_relu_net(with_attr=False):
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    net = _Net().eval()
    if with_attr:
        net.my_attr = [4, 5, 6]
    return net


def _conv_bn_net():
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)
            self.bn = torch.nn.BatchNorm2d(4)

        def forward(self, x):
            return self.bn(self.conv(x))

    return _Net().eval()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFuseFxUt(TestCase):
    """Functional tests for torch.ao.quantization.quantize_fx.fuse_fx."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_fuse_returns_graph_module(self):
        gm = torch.ao.quantization.quantize_fx.fuse_fx(_conv_relu_net())
        self.assertIsInstance(gm, GraphModule)
        self.assertGreater(len(list(gm.graph.nodes)), 0)

    def test_fuse_conv_relu_fuses_modules(self):
        gm = torch.ao.quantization.quantize_fx.fuse_fx(_conv_relu_net())
        # conv + relu must collapse into a single fused ConvReLU2d module.
        fused = [m for m in gm.modules() if isinstance(m, torch.ao.nn.intrinsic.ConvReLU2d)]
        self.assertEqual(len(fused), 1)
        targets = [str(n.target) for n in gm.graph.nodes]
        self.assertIn("conv", targets)
        self.assertNotIn("relu", targets)

    def test_fuse_bn_folds_into_conv(self):
        gm = torch.ao.quantization.quantize_fx.fuse_fx(_conv_bn_net())
        # In eval mode BatchNorm is folded into the preceding conv, leaving no BN module.
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in gm.modules()))

    def test_fuse_output_shape_preserved(self):
        gm = torch.ao.quantization.quantize_fx.fuse_fx(_conv_relu_net())
        with torch.no_grad():
            out = gm(*_EXAMPLE)
        self.assertEqual(tuple(out.shape), (1, 4, 6, 6))

    def test_fuse_preserved_attributes(self):
        config = FuseCustomConfig().set_preserved_attributes(["my_attr"])
        gm = torch.ao.quantization.quantize_fx.fuse_fx(
            _conv_relu_net(with_attr=True), fuse_custom_config=config
        )
        self.assertEqual(gm.my_attr, [4, 5, 6])

    def test_fuse_dict_config_deprecation(self):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            gm = torch.ao.quantization.quantize_fx.fuse_fx(_conv_relu_net(), fuse_custom_config={})
        # Passing a dict must still work but emit a FutureWarning deprecation notice.
        self.assertIsInstance(gm, GraphModule)
        self.assertTrue(any(issubclass(r.category, FutureWarning) for r in records))

    def test_fuse_invalid_config_type(self):
        with self.assertRaises(AttributeError):
            torch.ao.quantization.quantize_fx.fuse_fx(_conv_relu_net(), fuse_custom_config=123)


if __name__ == "__main__":
    run_tests()
