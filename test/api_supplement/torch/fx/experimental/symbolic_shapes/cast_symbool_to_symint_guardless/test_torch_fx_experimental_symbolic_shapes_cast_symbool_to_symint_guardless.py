# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.cast_symbool_to_symint_guardless 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.cast_symbool_to_symint_guardless
API 签名：cast_symbool_to_symint_guardless(symbool) -> int | torch.SymInt

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：场景所需的 Python 标量、真实 SymInt/SymBool、FX 节点或缓存回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：符号布尔转换无新增 guard |
| 正常传参场景 | 已覆盖：True/False 到 1/0 的精确类型和值 |
| 异常传参场景 | 不适用：有效符号输入的确定结果 |
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

from torch.fx.experimental.symbolic_shapes import cast_symbool_to_symint_guardless, ShapeEnv


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


    def test_cast_symbool_to_symint_guardless_python_boolean_maps_to_exact_integer(self):
        self.assertEqual(cast_symbool_to_symint_guardless(True), 1)
        self.assertEqual(cast_symbool_to_symint_guardless(False), 0)
        self.assertIs(type(cast_symbool_to_symint_guardless(True)), int)

    def test_cast_symbool_to_symint_guardless_symbolic_conversion_keeps_expression_and_no_guards(self):
        env = self._env()
        value = env.create_unbacked_symint()
        before = list(env.guards)
        converted = cast_symbool_to_symint_guardless(value > 3)
        self.assertIsInstance(converted, torch.SymInt)
        self.assertEqual(converted.node.expr.subs(value.node.expr, 5), 1)
        self.assertEqual(converted.node.expr.subs(value.node.expr, 2), 0)
        self.assertEqual(env.guards, before)


if __name__ == "__main__":
    run_tests()
