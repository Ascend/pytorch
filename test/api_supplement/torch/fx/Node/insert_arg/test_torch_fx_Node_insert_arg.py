# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.Node.insert_arg 接口功能正确性
API 名称：torch.fx.Node.insert_arg
API 签名：Node.insert_arg(self, idx, arg) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空参数列表、None、空字符串及空容器 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | Node、容器、Tensor、符号标量、数值/字符串、range/slice、dtype/device/layout 等 Argument |
| 传参与不传参 | 不适用：idx 和 arg 均必传 |
| 等价类/边界值 | 头/尾插入、重复依赖、值/身份保持及非法索引 |
| 正常传参场景 | 头/尾插入、嵌套 Node 参数和 users 更新 |
| 异常传参场景 | 负索引及越界索引不修改原参数 |
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




@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_insert_arg_positions_and_user_bookkeeping(self):
        graph = torch.fx.Graph()
        left = graph.placeholder("left")
        right = graph.placeholder("right")
        node = graph.call_function(torch.add, (left,))
        node.insert_arg(1, right)
        self.assertEqual(node.args, (left, right))
        self.assertIn(node, right.users)
        self.assertIn(node, left.users)
        self.assertEqual(node.all_input_nodes, [left, right])
        node.insert_arg(0, 7)
        node.insert_arg(len(node.args), {"nested": [right, left]})
        self.assertEqual(node.args, (7, left, right, {"nested": [right, left]}))
        self.assertEqual(len(left.users), 1)
        self.assertEqual(len(right.users), 1)

    def test_insert_arg_empty_arguments_and_invalid_index_leave_state_intact(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        node = graph.call_function(tuple, ())
        node.insert_arg(0, [source])
        self.assertEqual(node.args, ([source],))
        before = node.args
        for index in (-1, 2):
            with self.subTest(index=index):
                with self.assertRaises(AssertionError):
                    node.insert_arg(index, source)
                self.assertEqual(node.args, before)
                self.assertEqual(list(source.users), [node])


    def test_parameter_all_argument_families_preserve_values_and_nested_node_users(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        graph = torch.fx.Graph()
        source = graph.placeholder("value")
        symint = env.create_unbacked_symint()
        symfloat = env.create_unbacked_symfloat()
        symbool = symint > 0
        tensor = torch.ones(1, device=self.device)
        values = (None, False, 0, -2, 0.5, 1 + 2j, "", "name", torch.float32, self.device,
                  torch.contiguous_format, torch.strided, torch.ops.aten.neg.default, tensor,
                  symint, symfloat, symbool, (), [], {}, range(3), slice(0, 2), source,
                  (source,), [source], {"value": source})
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                node = graph.call_function(tuple, ())
                node.insert_arg(0, value)
                self.assertEqual(len(node.args), 1)
                observed = node.args[0]
                if isinstance(value, (list, tuple, dict)):
                    self.assertEqual(observed, value)
                else:
                    self.assertIs(observed, value)
                contains_node = value is source or isinstance(value, (tuple, list, dict)) and bool(value)
                self.assertEqual(node in source.users, contains_node)


if __name__ == "__main__":
    run_tests()
