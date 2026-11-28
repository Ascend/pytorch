# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.update_in 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.update_in
API 签名：update_in(d, keys, func, default=None, factory=dict) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空与非空输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | dict/MappingProxyType、list/tuple/迭代器路径、回调和任意默认值 |
| 传参与不传参 | default/factory 省略及显式 None/对象/dict/OrderedDict |
| 等价类/边界值 | 缺失路径、None 叶子、默认值身份和嵌套复制 |
| 正常传参场景 | 回调更新、嵌套复制、默认值及工厂 |
| 异常传参场景 | 空路径 StopIteration；None 路径/回调/工厂 TypeError |
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
from torch.fx.experimental.unification.unification_tools import update_in


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_update_in_nested_copy_and_callback(self):
        original = {"a": {"b": 4, "keep": 8}, "keep": 2}
        actual = update_in(original, ["a", "b"], lambda value: value * 3)
        self.assertEqual(actual, {"a": {"b": 12, "keep": 8}, "keep": 2})
        self.assertEqual(original, {"a": {"b": 4, "keep": 8}, "keep": 2})

    def test_update_in_missing_default_and_factory(self):
        actual = update_in({}, ("a", "b"), lambda value: value + 2, default=5, factory=OrderedDict)
        self.assertIsInstance(actual, OrderedDict)
        self.assertIsInstance(actual["a"], OrderedDict)
        self.assertEqual(actual, {"a": {"b": 7}})
        self.assertEqual(update_in({}, ["x"], lambda value: value is None), {"x": True})

    def test_update_in_empty_key_path(self):
        with self.assertRaises(StopIteration):
            update_in({}, [], lambda value: value)


    def test_parameter_mapping_protocol_iterable_paths_and_none_leaf(self):
        from types import MappingProxyType

        payload = object()
        for form in (list, tuple, iter):
            original = {None: {"leaf": None, "keep": 2}}
            mapping = MappingProxyType(original)
            keys = form([None, "leaf"])
            seen = []
            result = update_in(mapping, keys, lambda old: (seen.append(old), payload)[1], default=None, factory=dict)
            self.assertIs(result[None]["leaf"], payload)
            self.assertEqual(original, {None: {"leaf": None, "keep": 2}})
            self.assertEqual(result[None]["keep"], 2)
            self.assertIsNot(result[None], original[None])
            self.assertEqual(seen, [None])
        with self.assertRaises(TypeError):
            update_in({}, None, lambda old: old)

    def test_parameter_default_identity_and_none_callback_failure(self):
        for default in (None, [], {}, object()):
            seen = []

            def callback(value):
                seen.append(value)
                return value

            result = update_in({}, ["x"], callback, default=default, factory=dict)
            self.assertIs(seen[0], default)
            self.assertIs(result["x"], default)
        with self.assertRaises(TypeError):
            update_in({}, ["x"], None)
        with self.assertRaises(TypeError):
            update_in({}, ["x"], lambda value: value, factory=None)


if __name__ == "__main__":
    run_tests()
