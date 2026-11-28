# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.unification_tools.get_in 接口功能正确性
API 名称：torch.fx.experimental.unification.unification_tools.get_in
API 签名：get_in(keys, coll, default=None, no_default=False) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | None/标量/空集合，空/非空路径及默认值 |
| 枚举选项 | no_default=False/True |
| 参数类型 | 字典/列表/元组、list/tuple/迭代器路径、None 键和任意默认值 |
| 传参与不传参 | default/no_default 省略及显式 None/对象/布尔两值 |
| 等价类/边界值 | 缺失键、越界、空路径返回原集合及默认值身份 |
| 正常传参场景 | 字典/列表混合访问、默认值和严格模式 |
| 异常传参场景 | 严格模式 KeyError/IndexError/TypeError；None 路径的两种模式 |
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

from torch.fx.experimental.unification.unification_tools import get_in


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_get_in_nested_values_and_empty_path(self):
        original = {"a": {"items": [2, 7], "zero": 0}}
        self.assertEqual(get_in(["a", "items", 1], original), 7)
        self.assertEqual(get_in(("a", "zero"), original, default=99), 0)
        self.assertIs(get_in([], original), original)

    def test_get_in_missing_default_and_strict_errors(self):
        self.assertIsNone(get_in(["missing"], {}))
        self.assertEqual(get_in(["missing"], {}, default=17), 17)
        self.assertEqual(get_in([4], [1], default=23), 23)
        with self.assertRaises(KeyError):
            get_in(["missing"], {}, no_default=True)
        with self.assertRaises(IndexError):
            get_in([4], [1], no_default=True)
        with self.assertRaises(TypeError):
            get_in(["a", "b"], {"a": 1}, no_default=True)


    def test_parameter_none_collections_default_identity_and_explicit_false(self):
        for collection in (None, 0, [], {}, ()):
            for default in (None, [], {}, object()):
                with self.subTest(collection=collection, default_type=type(default).__name__):
                    self.assertIs(get_in(["missing"], collection, default=default, no_default=False), default)
                    self.assertIs(get_in(iter([]), collection, default=default, no_default=True), collection)
        self.assertIsNone(get_in([None], {None: None}, default=9, no_default=True))
        for strict in (False, True):
            if strict:
                with self.assertRaises(TypeError):
                    get_in(None, {}, no_default=strict)
            else:
                self.assertIsNone(get_in(None, {}, no_default=strict))


if __name__ == "__main__":
    run_tests()
