# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.keyfilter 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.keyfilter
API 签名：keyfilter(predicate, d, factory=dict) -> dict[_K, _V]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空/非空映射；键和值包含 None 和容器 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | dict/MappingProxyType、函数/可调用对象和 dict/OrderedDict 工厂 |
| 传参与不传参 | factory 省略及显式 dict/OrderedDict/None |
| 等价类/边界值 | 空映射不调用回调；逐项转换/过滤、值身份及原映射保持 |
| 正常传参场景 | 过滤结果、保留原输入与显式工厂 |
| 异常传参场景 | 非空输入的 None 回调、None 映射和 None 工厂错误 |
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
from torch.fx.experimental.unification.unification_tools import keyfilter


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_keyfilter_filter_retains_only_matches(self):
        original = {1: 2, 2: 3, 3: 5, 4: 7}
        actual = keyfilter(lambda key: key % 2 == 0, original)
        self.assertEqual(actual, {2: 3, 4: 7})
        self.assertEqual(original, {1: 2, 2: 3, 3: 5, 4: 7})

    def test_keyfilter_empty_and_factory(self):
        self.assertEqual(keyfilter(lambda value: True, {}), {})
        result = keyfilter(lambda value: True, {"a": 0, "b": None}, factory=OrderedDict)
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(list(result.items()), [("a", 0), ("b", None)])


    def test_parameter_mapping_and_callable_protocols_empty_none_and_explicit_factory(self):
        from types import MappingProxyType
        from collections import OrderedDict

        calls = []

        class Callback:
            def __call__(self, value):
                calls.append(value)
                return True

        original = {None: None, "list": [], (1, 2): {"a": 3}}
        result = keyfilter(Callback(), MappingProxyType(original), factory=OrderedDict)
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(list(result), list(original))
        for key in original:
            self.assertIs(result[key], original[key])
        self.assertEqual(len(calls), 3)
        self.assertEqual(keyfilter(None, {}, factory=dict), {})
        with self.assertRaises(TypeError):
            keyfilter(None, {"key": 1})
        with self.assertRaises(AttributeError):
            keyfilter(Callback(), None)
        with self.assertRaises(TypeError):
            keyfilter(Callback(), {}, factory=None)


if __name__ == "__main__":
    run_tests()
