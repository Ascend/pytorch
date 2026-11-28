# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.optimization.remove_dropout 接口功能正确性
API 名称：torch.fx.experimental.optimization.remove_dropout
API 签名：remove_dropout(model) -> torch.nn.modules.module.Module

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空节点集合或无匹配状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 图、模块、schema 及接口默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无 Dropout、空张量与两组不同输入 |
| 正常传参场景 | 已覆盖：移除多个 Dropout、保留 ReLU 并在设备执行 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 未覆盖：单设备模型不涉及混合设备输入 |

未覆盖项及原因：
- 图执行方式：直接执行改写后的 FX GraphModule，未覆盖 Inductor codegen。
- eager 对照：相同设备和输入下比较 eval 模型结果，包括两组不同输入。
- autograd：未覆盖 backward；本用例验证推理图改写。
- RNG：eval Dropout 无随机性，使用确定输入。
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

from torch.fx.experimental.optimization import remove_dropout


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_remove_dropout_dropout_nodes_are_removed_and_relu_result_is_preserved(self):
        self.assertIsNotNone(self.device)
        model = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.ReLU(), torch.nn.Dropout(0.2)).eval()
        transformed = remove_dropout(model)
        calls = [transformed.get_submodule(node.target) for node in transformed.graph.nodes
                 if node.op == "call_module"]
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in calls))
        self.assertTrue(any(isinstance(module, torch.nn.ReLU) for module in calls))
        for shift in (0.0, 2.0):
            value = torch.arange(-3, 3, dtype=torch.float32, device=self.device).reshape(2, 3) + shift
            result = transformed(value)
            torch.testing.assert_close(result.cpu(), model(value).cpu())
            self.assertEqual(result.device.type, self.device_name)

    def test_remove_dropout_no_dropout_and_empty_input_preserve_values(self):
        self.assertIsNotNone(self.device)
        model = torch.nn.Sequential(torch.nn.ReLU()).eval()
        transformed = remove_dropout(model)
        for shape in ((0, 3), (2, 3)):
            value = torch.full(shape, -2.0, device=self.device)
            result = transformed(value)
            torch.testing.assert_close(result.cpu(), model(value).cpu())
            self.assertEqual(result.shape, value.shape)
            self.assertEqual(result.dtype, value.dtype)


if __name__ == "__main__":
    run_tests()
