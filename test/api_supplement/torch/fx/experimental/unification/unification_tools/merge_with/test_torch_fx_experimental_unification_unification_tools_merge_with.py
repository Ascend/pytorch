# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.merge_with 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.merge_with
API 签名：merge_with(func, *dicts, **kwargs) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空与非空输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | dict/OrderedDict/MappingProxyType，list/tuple/迭代器入口和 None 值 |
| 传参与不传参 | 零/单/多映射；factory 省略及显式 OrderedDict/None |
| 等价类/边界值 | 空映射、同键合并顺序、输入保持和有序工厂 |
| 正常传参场景 | 同键值聚合、顺序、序列入口和工厂 |
| 异常传参场景 | None 映射/工厂和未知关键字 TypeError |
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

from collections import OrderedDict
from torch.fx.experimental.unification.unification_tools import merge_with


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_merge_with_combines_all_values_without_mutation(self):
        first_mapping = {"x": 2, "keep": 3}
        second_mapping = {"x": 7, "new": 4}
        self.assertEqual(merge_with(sum, first_mapping, second_mapping), {"x": 9, "keep": 3, "new": 4})
        self.assertEqual(first_mapping, {"x": 2, "keep": 3})
        self.assertEqual(second_mapping, {"x": 7, "new": 4})

    def test_merge_with_order_empty_iterable_and_factory(self):
        self.assertEqual(merge_with(tuple, [{"x": 2}, {"x": 7}]), {"x": (2, 7)})
        self.assertEqual(merge_with(sum), {})
        result = merge_with(sum, [{}, {"x": 0}], factory=OrderedDict)
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(list(result.items()), [("x", 0)])


    def test_parameter_mapping_protocol_iterable_forms_and_invalid_options(self):
        from types import MappingProxyType
        from collections import OrderedDict

        mappings = [MappingProxyType({None: None, "x": 1}), OrderedDict(x=2)]
        for form in (list, tuple, iter):
            result = merge_with(tuple, form(mappings), factory=OrderedDict)
            self.assertIsInstance(result, OrderedDict)
            self.assertEqual(result, {None: (None,), "x": (1, 2)})
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            merge_with(tuple, unexpected=True)
        with self.assertRaises(TypeError):
            merge_with(tuple, {}, factory=None)
        with self.assertRaises(TypeError):
            merge_with(tuple, None)


if __name__ == "__main__":
    run_tests()
