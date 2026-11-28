# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unification.variable.var 接口功能正确性
API 名称：torch.fx.experimental.unification.variable.var
API 签名：var() -> Callable[..., Var]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空输入或不匹配分支；无容器入参时不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：本文件列出的具体类型、值和回调组合 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零 token 和多个 token |
| 正常传参场景 | 已覆盖：当前版本工厂接口的具名变量语义 |
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

from torch.fx.experimental.unification.variable import var
from torch.fx.experimental.unification.variable import Var


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_var_factory_creates_named_variables(self):
        factory = var()
        self.assertEqual(factory("x"), Var("x"))
        self.assertNotEqual(factory("x"), factory("y"))
        self.assertEqual(factory("x", "y"), Var("x", "y"))

    def test_var_anonymous_tokens_are_distinct(self):
        previous = Var._id
        self.addCleanup(setattr, Var, "_id", previous)
        factory = var()
        first, second = factory(), factory()
        self.assertNotEqual(first, second)
        self.assertEqual(Var._id, previous + 2)


if __name__ == "__main__":
    run_tests()
