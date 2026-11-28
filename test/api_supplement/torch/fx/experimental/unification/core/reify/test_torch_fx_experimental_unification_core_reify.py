# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.core.reify 接口功能正确性
API 名称：torch.fx.experimental.unification.core.reify
API 签名：reify(e, s) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空替换表/容器、None 和空字符串 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | Var、dict/list/tuple、惰性迭代器和普通对象 |
| 传参与不传参 | 不适用：e 和 s 均必传 |
| 等价类/边界值 | 传递替换、未绑定变量、惰性消费和普通值身份 |
| 正常传参场景 | 嵌套和传递替换的具体值 |
| 异常传参场景 | None 替换表 NotImplementedError |
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

from torch.fx.experimental.unification.core import reify
from torch.fx.experimental.unification.variable import Var


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_reify_nested_substitution_and_transitive_chain(self):
        x, y = Var("x"), Var("y")
        original = {"a": (x, [y, 9])}
        self.assertEqual(reify(original, {x: y, y: 4}), {"a": (4, [4, 9])})
        self.assertEqual(original, {"a": (x, [y, 9])})

    def test_reify_unbound_variables_and_empty_mapping(self):
        x = Var("x")
        self.assertIs(reify(x, {}), x)
        self.assertEqual(reify([], {}), [])
        self.assertEqual(reify({"x": x}, {}), {"x": x})


    def test_parameter_iterator_is_lazy_and_ordinary_values_keep_identity(self):
        variable = Var("iterator_value")
        visited = []

        def values():
            for value in (variable, [variable], None):
                visited.append(value)
                yield value

        result = reify(values(), {variable: 7})
        self.assertEqual(visited, [])
        self.assertEqual(next(result), 7)
        self.assertEqual(len(visited), 1)
        self.assertEqual(list(result), [[7], None])
        for value in (None, True, 0, 0.5, "", frozenset({1}), object()):
            self.assertIs(reify(value, {}), value)
        with self.assertRaises(NotImplementedError):
            reify([], None)


if __name__ == "__main__":
    run_tests()
