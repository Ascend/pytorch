# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.assoc 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.assoc
API 签名：assoc(d, key, value, factory=dict) -> dict[_K, _V]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空/非空映射；None/空/非空键值 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | dict/OrderedDict/MappingProxyType；None/int/str/tuple 键和任意值对象 |
| 传参与不传参 | factory 省略及显式 dict/OrderedDict/None |
| 等价类/边界值 | 新增/覆盖、零值、值身份和原映射保持 |
| 正常传参场景 | 新增、覆盖及原映射不变 |
| 异常传参场景 | 不可哈希键、None 映射/工厂 TypeError |
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
from torch.fx.experimental.unification.unification_tools import assoc


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_assoc_copy_insert_and_overwrite(self):
        original = {"x": 3, "other": 9}
        self.assertEqual(assoc(original, "x", 7), {"x": 7, "other": 9})
        self.assertEqual(assoc(original, "new", None), {"x": 3, "other": 9, "new": None})
        self.assertEqual(original, {"x": 3, "other": 9})

    def test_assoc_empty_and_explicit_factory(self):
        result = assoc({}, "x", 0, factory=OrderedDict)
        self.assertIsInstance(result, OrderedDict)
        self.assertEqual(list(result.items()), [("x", 0)])


    def test_parameter_mapping_protocol_hashable_keys_and_arbitrary_value_identity(self):
        from types import MappingProxyType
        from collections import OrderedDict

        for mapping in ({}, OrderedDict(), MappingProxyType({})):
            for key in (None, 0, "", (1, 2)):
                for value in (None, [], {}, (1,), object()):
                    with self.subTest(mapping=type(mapping).__name__, key=key, value_type=type(value).__name__):
                        result = assoc(mapping, key, value, factory=dict)
                        self.assertIs(result[key], value)
                        self.assertEqual(len(result), 1)
                        self.assertEqual(len(mapping), 0)
        with self.assertRaises(TypeError):
            assoc({}, [], 1)
        with self.assertRaises(TypeError):
            assoc({}, "key", 1, factory=None)
        with self.assertRaises(TypeError):
            assoc(None, "key", 1)


if __name__ == "__main__":
    run_tests()
