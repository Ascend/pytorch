# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.splitter_base.generate_inputs_for_submodules 接口功能正确性
API 名称：torch.fx.passes.splitter_base.generate_inputs_for_submodules
API 签名：generate_inputs_for_submodules(model, inputs, target_submodules, deepcopy=False) -> dict[str, typing.Any]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空目标、深拷贝独立及异常退出 |
| 正常传参场景 | 已覆盖：真实模块前向输入记录和正常 hook 清理 |
| 异常传参场景 | 已覆盖：模块异常原样传播及 hook 移除 |
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

from torch.fx.passes.splitter_base import generate_inputs_for_submodules


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_generate_inputs_for_submodules_inputs_are_captured_and_hooks_removed(self):
        class Triple(torch.nn.Module):
            def forward(self, value):
                return value * 3

        model = torch.nn.Sequential(Triple(), torch.nn.Identity())
        before = [len(module._forward_pre_hooks) for module in model.modules()]
        result = generate_inputs_for_submodules(model, (4,), ["0", "1"])
        self.assertEqual(result, {"0": (4,), "1": (12,)})
        self.assertEqual([len(module._forward_pre_hooks) for module in model.modules()], before)
        self.assertEqual(generate_inputs_for_submodules(model, (2,), []), {})

    def test_generate_inputs_for_submodules_deepcopy_is_independent_and_error_exit_removes_hooks(self):
        model = torch.nn.Sequential(torch.nn.Identity())
        value = [2, 3]
        result = generate_inputs_for_submodules(model, (value,), ["0"], deepcopy=True)
        value.append(4)
        self.assertEqual(result["0"], ([2, 3],))
        self.assertIsNot(result["0"][0], value)

        class Failure(torch.nn.Module):
            def forward(self, value):
                raise RuntimeError("controlled module error")

        failing = torch.nn.Sequential(torch.nn.Identity(), Failure())
        before = [len(module._forward_pre_hooks) for module in failing.modules()]
        with self.assertRaisesRegex(RuntimeError, "controlled module error"):
            generate_inputs_for_submodules(failing, (5,), ["0", "1"])
        self.assertEqual([len(module._forward_pre_hooks) for module in failing.modules()], before)


if __name__ == "__main__":
    run_tests()
