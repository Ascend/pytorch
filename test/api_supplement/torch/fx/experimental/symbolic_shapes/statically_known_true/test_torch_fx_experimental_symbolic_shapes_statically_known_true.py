# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.statically_known_true 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.statically_known_true
API 签名：statically_known_true(x) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：场景所需的 Python 标量、真实 SymInt/SymBool、FX 节点或缓存回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：未知谓词不安装 guard |
| 正常传参场景 | 已覆盖：真/假常量和可化简符号恒真式 |
| 异常传参场景 | 已覆盖：非 bool/SymBool 输入抛出 AssertionError |
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

from torch.fx.experimental.symbolic_shapes import statically_known_true, ShapeEnv


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


    def test_statically_known_true_constants_and_symbolic_tautology(self):
        self.assertTrue(statically_known_true(True))
        self.assertFalse(statically_known_true(False))
        env = self._env()
        value = env.create_unbacked_symint()
        self.assertTrue(statically_known_true(value == value))
        self.assertTrue(statically_known_true(value + 1 > value))

    def test_statically_known_true_unknown_predicate_does_not_add_guards(self):
        env = self._env()
        value = env.create_unbacked_symint()
        before = list(env.guards)
        self.assertFalse(statically_known_true(value > 3))
        self.assertEqual(env.guards, before)
        with self.assertRaises(AssertionError):
            statically_known_true(1)


if __name__ == "__main__":
    run_tests()
