# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.optimization_hint 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.optimization_hint
API 签名：optimization_hint(a, fallback=None) -> int

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：场景所需的 Python 标量、真实 SymInt/SymBool、FX 节点或缓存回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无 hint fallback 不安装 guard |
| 正常传参场景 | 已覆盖：具体和带 hint 的表达式优化提示 |
| 异常传参场景 | 已覆盖：bool 不当作整数优化提示接受 |
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

from torch.fx.experimental.symbolic_shapes import optimization_hint, ShapeEnv
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _env(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        return env


    def test_optimization_hint_concrete_and_backed_expression_hints(self):
        self.assertEqual(optimization_hint(-3), -3)
        env = self._env()
        value = env.create_unspecified_symint_and_symbol(5, LocalSource("value"), DimDynamic.DYNAMIC)
        self.assertEqual(optimization_hint(value + 2), 7)
        self.assertEqual(env.guards, [])

    def test_optimization_hint_unbacked_fallback_is_non_guarding(self):
        env = self._env()
        value = env.create_unbacked_symint()
        self.assertEqual(optimization_hint(value, fallback=11), 11)
        self.assertEqual(env.guards, [])
        with self.assertRaises(AssertionError):
            optimization_hint(True)


if __name__ == "__main__":
    run_tests()
