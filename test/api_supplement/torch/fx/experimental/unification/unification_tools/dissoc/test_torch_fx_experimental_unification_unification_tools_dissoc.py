# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.dissoc 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.dissoc
API 签名：dissoc(d, *keys, **kwargs) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空与非空输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | dict/MappingProxyType、None/字符串/元组键及 dict/OrderedDict 工厂 |
| 传参与不传参 | 零/单/多删除键；factory 省略及显式 dict/OrderedDict/None |
| 等价类/边界值 | 稀疏/密集删除、重复/缺失键及原映射保持 |
| 正常传参场景 | 两种删除分支及原映射不变 |
| 异常传参场景 | 未知关键字及 None 工厂 TypeError |
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
from torch.fx.experimental.unification.unification_tools import dissoc


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_dissoc_sparse_and_dense_removal(self):
        original = {"a": 1, "b": 2, "c": 3, "d": 4}
        self.assertEqual(dissoc(original, "a"), {"b": 2, "c": 3, "d": 4})
        self.assertEqual(dissoc(original, "a", "b", "c"), {"d": 4})
        self.assertEqual(original, {"a": 1, "b": 2, "c": 3, "d": 4})

    def test_dissoc_empty_missing_and_no_keys(self):
        self.assertEqual(dissoc({}, "missing"), {})
        self.assertEqual(dissoc({"x": None}, "missing"), {"x": None})
        result = dissoc({"x": 1}, factory=OrderedDict)
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(list(result.items()), [("x", 1)])


    def test_parameter_mapping_protocol_none_and_tuple_keys_factories_and_unknown_option(self):
        from types import MappingProxyType
        from collections import OrderedDict

        original = {None: 1, (2, 3): 4, "keep": None}
        for keys in ((), (None,), (None, (2, 3)), (None, None, "missing")):
            expected = {key: value for key, value in original.items() if key not in keys}
            for factory in (dict, OrderedDict):
                result = dissoc(MappingProxyType(original), *keys, factory=factory)
                self.assertIsInstance(result, factory)
                self.assertEqual(result, expected)
        self.assertEqual(original, {None: 1, (2, 3): 4, "keep": None})
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            dissoc({}, unexpected=True)
        with self.assertRaises(TypeError):
            dissoc({}, factory=None)


if __name__ == "__main__":
    run_tests()
