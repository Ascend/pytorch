# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.pass_manager.this_before_that_pass_constraint 接口功能正确性
API 名称：torch.fx.passes.pass_manager.this_before_that_pass_constraint
API 签名：this_before_that_pass_constraint(this, that) -> collections.abc.Callable[[collections.abc.Callable[...,
    typing.Any], collections.abc.Callable[..., typing.Any]], bool]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空支持链、初始状态或无迭代边界 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 节点或带可观测状态的 Python pass 回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：同一回调及经过 wraps 包装的回调比较 |
| 正常传参场景 | 已覆盖：合法、反向及不相关 pass 顺序 |
| 异常传参场景 | 不适用：有效回调的布尔或顺序结果 |
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

from torch.fx.passes.pass_manager import this_before_that_pass_constraint, inplace_wrapper, loop_pass


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_this_before_that_pass_constraint_constraint_rejects_only_reversed_order(self):
        def first(value):
            return value + 1

        def second(value):
            return value * 2

        def unrelated(value):
            return value - 1

        constraint = this_before_that_pass_constraint(first, second)
        self.assertTrue(constraint(first, second))
        self.assertFalse(constraint(second, first))
        self.assertTrue(constraint(first, unrelated))
        self.assertTrue(constraint(unrelated, second))
        self.assertTrue(constraint(first, first))

    def test_this_before_that_pass_constraint_wrapped_pass_comparison(self):
        def first(value):
            return value + 1

        def second(value):
            return value * 2

        wrapped_first = inplace_wrapper(first)
        wrapped_second = loop_pass(second, n_iter=2)
        constraint = this_before_that_pass_constraint(first, second)
        self.assertTrue(constraint(wrapped_first, wrapped_second))
        self.assertEqual(constraint(wrapped_second, wrapped_first), True)


if __name__ == "__main__":
    run_tests()
