# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint_generator.assert_inference_rule 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint_generator.assert_inference_rule
API 签名：assert_inference_rule(n, symbols, constraints, counter) ->
    tuple[list[torch.fx.experimental.migrate_gradual_types.constraint.Constraint], int]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：空 symbols/constraints |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 FX Node、输入类型符号及适用的 nn.Module/函数参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：结果被使用时显式拒绝 |
| 正常传参场景 | 已覆盖：无用户 assert 节点不产生形状约束 |
| 异常传参场景 | 已覆盖：assert 节点存在用户时抛出 AssertionError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 仅验证直接返回值及符号状态，不声明 SMT 形状推导或数值算子支持。
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

from torch.fx.experimental.migrate_gradual_types.constraint_generator import assert_inference_rule
from torch.fx.experimental.migrate_gradual_types.constraint_generator import _assert_is_none


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_assert_inference_rule_unused_assert_node_produces_no_shape_constraints(self):
        graph = torch.fx.Graph()
        node = graph.call_function(_assert_is_none, (None, "expected None"))
        symbols = {}
        self.assertEqual(assert_inference_rule(node, symbols, [], 17), ([], 17))
        self.assertEqual(symbols, {})

    def test_assert_inference_rule_assert_result_cannot_have_users(self):
        graph = torch.fx.Graph()
        node = graph.call_function(_assert_is_none, (None, "expected None"))
        graph.output(node)
        with self.assertRaisesRegex(AssertionError, "no users"):
            assert_inference_rule(node, {}, [], 17)


if __name__ == "__main__":
    run_tests()
