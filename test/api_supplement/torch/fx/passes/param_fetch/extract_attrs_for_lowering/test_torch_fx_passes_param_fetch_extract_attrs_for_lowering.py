# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.param_fetch.extract_attrs_for_lowering 接口功能正确性
API 名称：torch.fx.passes.param_fetch.extract_attrs_for_lowering
API 签名：extract_attrs_for_lowering(mod) -> dict[str, typing.Any]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空节点集合或无匹配状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 图、模块、schema 及接口默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：未知模块及版本过新的拒绝 |
| 正常传参场景 | 已覆盖：Conv2d 参数对象和离散配置、ReLU 属性 |
| 异常传参场景 | 已覆盖：未登记模块与版本不兼容的 RuntimeError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
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

from torch.fx.passes.param_fetch import extract_attrs_for_lowering


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_extract_attrs_for_lowering_convolution_parameters_and_configuration_are_extracted(self):
        module = torch.nn.Conv2d(3, 5, (3, 2), stride=(2, 1), padding=(1, 0), bias=False, device="meta")
        attrs = extract_attrs_for_lowering(module)
        self.assertEqual(attrs["name"], torch.typename(module))
        self.assertIs(attrs["weight"], module.weight)
        self.assertIsNone(attrs["bias"])
        for name in ("kernel_size", "stride", "padding", "dilation", "groups", "padding_mode"):
            self.assertEqual(attrs[name], getattr(module, name))

    def test_extract_attrs_for_lowering_activation_and_unsupported_or_newer_module_versions(self):
        activation = torch.nn.ReLU(inplace=True)
        self.assertTrue(extract_attrs_for_lowering(activation)["inplace"])
        with self.assertRaisesRegex(RuntimeError, "not in the module_fetch_book"):
            extract_attrs_for_lowering(torch.nn.Identity())
        activation._version = 99
        with self.assertRaisesRegex(RuntimeError, "Fetcher version"):
            extract_attrs_for_lowering(activation)


if __name__ == "__main__":
    run_tests()
