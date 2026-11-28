# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.node.map_arg 接口功能正确性
API 名称：torch.fx.node.map_arg
API 签名：map_arg(a, fn) -> ~ArgumentT

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | None、空 list/tuple/dict 及空字符串 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | Node、list/tuple/dict/slice、Tensor、符号标量和设备/布局等 Argument |
| 传参与不传参 | 不适用：a 和 fn 均必传 |
| 等价类/边界值 | 重复 Node、空容器；非 Node 值保持身份且不调用回调 |
| 正常传参场景 | list/tuple/dict/slice 中递归替换 Node |
| 异常传参场景 | 非 callable 回调抛出 AssertionError |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

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

from torch.fx.node import map_arg


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_map_arg_nested_structure_transforms_only_nodes(self):
        graph = torch.fx.Graph()
        left = graph.placeholder("left")
        right = graph.placeholder("right")
        arguments = {"values": [left, (right, 7)], "slice": slice(left, right, 2), "none": None}
        result = map_arg(arguments, lambda node: node.name)
        self.assertEqual(result, {"values": ["left", ("right", 7)],
                                  "slice": slice("left", "right", 2), "none": None})
        self.assertIs(arguments["values"][0], left)
        self.assertIs(arguments["values"][1][0], right)

    def test_map_arg_repeated_nodes_empty_containers_and_invalid_callback(self):
        node = torch.fx.Graph().placeholder("x")
        visited = []

        def replace(current):
            visited.append(current)
            return 11

        self.assertEqual(map_arg((node, node, [], {}), replace), (11, 11, [], {}))
        self.assertEqual(visited, [node, node])
        with self.assertRaises(AssertionError):
            map_arg(node, None)


    def test_parameter_scalar_argument_families_pass_through_without_callback(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        scalar = env.create_unbacked_symint()
        values = (None, False, 0, -2, 0.5, 1 + 2j, "", torch.float32, self.device,
                  torch.contiguous_format, torch.strided, torch.ops.aten.neg.default,
                  torch.ones(1, device=self.device), scalar, env.create_unbacked_symfloat(), scalar > 0, range(3))
        calls = []
        for value in values:
            self.assertIs(map_arg(value, lambda node: calls.append(node)), value)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    run_tests()
