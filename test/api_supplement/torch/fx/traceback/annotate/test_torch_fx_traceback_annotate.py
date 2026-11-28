# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.traceback.annotate 接口功能正确性
API 名称：torch.fx.traceback.annotate
API 签名：annotate(annotation_dict) -> collections.abc.Iterator[None]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：嵌套覆盖、异常恢复和两组输入 |
| 正常传参场景 | 已覆盖：PT2 export 节点注解和设备图执行 |
| 异常传参场景 | 已覆盖：受控异常后原注解恢复 |
| 混合设备类型 | 未覆盖：单 Tensor 无混合设备语义 |

未覆盖项及原因：
- 图执行方式：torch.export.export(strict=True) 捕获真实 ATen 图；未覆盖 Inductor codegen。
- eager 对照：同设备、同输入，两组不同输入验证 export 模块结果。
- autograd：未覆盖梯度累加注解；该接口文档排除该场景。
- RNG：使用确定输入，无随机流。
- 参数类型：非严格 export 路径不保留本版本 custom 元数据，本文件只验证严格 PT2 路径。
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

import copy
from torch.fx import traceback as fx_traceback


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_annotate_nested_annotations_and_exception_restore_context(self):
        with fx_traceback.preserve_node_meta():
            original = copy.deepcopy(fx_traceback.get_current_meta())
            with fx_traceback.annotate({"tag": "outer", "number": 7}):
                self.assertEqual(fx_traceback.get_current_meta()["custom"]["tag"], "outer")
                with self.assertRaisesRegex(RuntimeError, "controlled"):
                    with fx_traceback.annotate({"tag": "inner"}):
                        self.assertEqual(fx_traceback.get_current_meta()["custom"]["tag"], "inner")
                        self.assertEqual(fx_traceback.get_current_meta()["custom"]["number"], 7)
                        raise RuntimeError("controlled")
                self.assertEqual(fx_traceback.get_current_meta()["custom"]["tag"], "outer")
            self.assertEqual(fx_traceback.get_current_meta(), original)

    def test_annotate_export_nodes_carry_annotations_and_execute_on_selected_device(self):
        self.assertIsNotNone(self.device)
        self.addCleanup(torch._dynamo.reset)

        class Annotated(torch.nn.Module):
            def forward(self, value):
                with fx_traceback.annotate({"tag": "ut_export", "stage": 7}):
                    return value.sin() + 2

        model = Annotated()
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 5
        program = torch.export.export(model, (value,), strict=True)
        calls = [node for node in program.graph_module.graph.nodes if node.op == "call_function"]
        self.assertTrue(calls)
        for node in calls:
            self.assertEqual(node.meta["custom"], {"tag": "ut_export", "stage": 7})
        executable = program.module()
        for current in (value, value + 0.4):
            result = executable(current)
            torch.testing.assert_close(result.cpu(), model(current).cpu())
            self.assertEqual(result.shape, current.shape)
            self.assertEqual(result.dtype, current.dtype)
            self.assertEqual(result.device.type, self.device_name)


if __name__ == "__main__":
    run_tests()
