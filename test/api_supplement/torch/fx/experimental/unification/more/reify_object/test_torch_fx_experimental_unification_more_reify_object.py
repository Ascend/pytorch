# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.more.reify_object 接口功能正确性
API 名称：torch.fx.experimental.unification.more.reify_object
API 签名：reify_object(o, s) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空输入或不匹配分支；无容器入参时不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：本文件列出的具体类型、值和回调组合 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空替换复用原对象 |
| 正常传参场景 | 已覆盖：普通对象与 slots 对象字段替换 |
| 异常传参场景 | 未覆盖：没有声明额外参数校验 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

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

from torch.fx.experimental.unification.more import reify_object
from torch.fx.experimental.unification.variable import Var


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_reify_object_dictionary_object_substitution_preserves_original(self):
        class Term:
            def __init__(self, value):
                self.value = value
                self.keep = 7

        variable = Var("x")
        original = Term(variable)
        actual = reify_object(original, {variable: 4})
        self.assertEqual((actual.value, actual.keep), (4, 7))
        self.assertEqual((original.value, original.keep), (variable, 7))
        self.assertIs(reify_object(original, {}), original)

    def test_reify_object_slotted_object_substitution(self):
        class Term:
            __slots__ = ("value", "keep")

            def __init__(self, value):
                self.value, self.keep = value, 8

        variable = Var("x")
        original = Term(variable)
        actual = reify_object(original, {variable: 3})
        self.assertEqual((actual.value, actual.keep), (3, 8))
        self.assertEqual(original.value, variable)


if __name__ == "__main__":
    run_tests()
