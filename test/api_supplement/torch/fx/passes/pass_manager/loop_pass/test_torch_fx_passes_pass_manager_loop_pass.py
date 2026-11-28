# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.pass_manager.loop_pass 接口功能正确性
API 名称：torch.fx.passes.pass_manager.loop_pass
API 签名：loop_pass(base_pass, n_iter=None, predicate=None) -> collections.abc.Callable[[~_T], ~_T]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空支持链、初始状态或无迭代边界 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 节点或带可观测状态的 Python pass 回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：初始 predicate 为假与非正次数 |
| 正常传参场景 | 已覆盖：固定迭代次数和基于新输出的 predicate 重检 |
| 异常传参场景 | 已覆盖：未提供/同时提供配置的 AssertionError 及非正次数 RuntimeError |
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

from torch.fx.passes.pass_manager import loop_pass


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_loop_pass_fixed_iterations_and_predicate_recheck(self):
        visited = []

        def advance(value):
            visited.append(value)
            return value + 2

        self.assertEqual(loop_pass(advance, n_iter=3)(1), 7)
        self.assertEqual(visited, [1, 3, 5])
        visited.clear()
        self.assertEqual(loop_pass(advance, predicate=lambda value: value < 6)(0), 6)
        self.assertEqual(visited, [0, 2, 4])
        visited.clear()
        self.assertEqual(loop_pass(advance, predicate=lambda value: False)(9), 9)
        self.assertEqual(visited, [])

    def test_loop_pass_invalid_loop_configuration(self):
        def identity(value):
            return value

        with self.assertRaises(AssertionError):
            loop_pass(identity)
        with self.assertRaises(AssertionError):
            loop_pass(identity, n_iter=2, predicate=lambda value: False)
        for count in (0, -1):
            with self.subTest(count=count):
                with self.assertRaisesRegex(RuntimeError, "positive int"):
                    loop_pass(identity, n_iter=count)(3)


if __name__ == "__main__":
    run_tests()
