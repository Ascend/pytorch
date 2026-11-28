# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.runtime_assert.insert_deferred_runtime_asserts 接口功能正确性
API 名称：torch.fx.passes.runtime_assert.insert_deferred_runtime_asserts
API 签名：insert_deferred_runtime_asserts(gm, shape_env, name, export=False) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 未覆盖：本文件使用非空约束图 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv/GraphModule、name 和 export 参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：export 开/关、合法/非法输入和用户消息去重 |
| 正常传参场景 | 已覆盖：真实未绑定符号延迟断言插入及执行 |
| 异常传参场景 | 已覆盖：运行时违反约束的 RuntimeError |
| 混合设备类型 | 未覆盖：单输入张量，无混合设备语义 |

未覆盖项及原因：
- 图执行方式：直接执行插入断言的 FX GraphModule；未覆盖 Inductor codegen。
- eager 对照：以确定标量关系和断言契约为基线，覆盖两组有效输入及非法输入。
- autograd：整数值与约束处理，不计算梯度。
- RNG：确定输入。
- 参数类型：未覆盖中间张量 stride/size CSE 及无法解释的 SymPy 函数。
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

import operator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _graph(self, env, user_check=False):
        size = env.create_unbacked_symint()
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        value.meta["val"] = torch.empty((), dtype=torch.int64, device="meta")
        item = graph.call_method("item", (value,))
        item.meta["val"] = size
        item.meta["unbacked_bindings"] = {size.node.expr: ()}
        root = torch.nn.Module()
        if user_check:
            condition = graph.call_function(operator.ge, (item, 2))
            condition.meta["val"] = size >= 2
            root.message = lambda: "minimum value is two"
            message = graph.create_node("get_attr", "message", (), {})
            message.meta["val"] = root.message
            check = graph.call_function(torch._check, (condition, message))
            check.meta["val"] = None
        result = graph.call_function(operator.add, (item, 1))
        result.meta["val"] = size + 1
        graph.output(result)
        return torch.fx.GraphModule(root, graph), size

    def test_insert_deferred_runtime_asserts_deferred_evenness_assert_is_inserted_and_executed(self):
        self.assertIsNotNone(self.device)
        for export in (False, True):
            with self.subTest(export=export):
                env = ShapeEnv()
                self.addCleanup(env.cleanup)
                module, size = self._graph(env)
                env.guard_or_defer_runtime_assert((size % 2 == 0).node.expr, "even input")
                before = {key: tuple(value) for key, value in env.deferred_runtime_asserts.items()}
                with FakeTensorMode(shape_env=env):
                    self.assertIsNone(insert_deferred_runtime_asserts(module, env, "ut_even", export=export))
                checks = [node for node in module.graph.nodes if node.target is torch.ops.aten._assert_scalar.default]
                self.assertEqual(len(checks), 1)
                # Evaluating inserted metadata can record the assertion again in ShapeEnv.
                for key, original in before.items():
                    for assertion in original:
                        self.assertIn(assertion, env.deferred_runtime_asserts[key])
                module.graph.lint()
                module.recompile()
                for value in (4, 8):
                    self.assertEqual(module(torch.tensor(value, device=self.device)), value + 1)
                with self.assertRaises(RuntimeError):
                    module(torch.tensor(3, device=self.device))

    def test_insert_deferred_runtime_asserts_user_assert_message_is_preserved_without_duplicate_bound_check(self):
        self.assertIsNotNone(self.device)
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        module, size = self._graph(env, user_check=True)
        env.guard_or_defer_runtime_assert(size.node.expr >= 2, "minimum value")
        with FakeTensorMode(shape_env=env):
            insert_deferred_runtime_asserts(module, env, "ut_user_bound")
        checks = [node for node in module.graph.nodes if node.target is torch.ops.aten._assert_scalar.default]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].args[1], "minimum value is two")
        self.assertFalse(any(node.target is torch._check for node in module.graph.nodes))
        module.graph.lint()
        module.recompile()
        for value in (2, 5):
            self.assertEqual(module(torch.tensor(value, device=self.device)), value + 1)
        with self.assertRaisesRegex(RuntimeError, "minimum value is two"):
            module(torch.tensor(1, device=self.device))


if __name__ == "__main__":
    run_tests()
