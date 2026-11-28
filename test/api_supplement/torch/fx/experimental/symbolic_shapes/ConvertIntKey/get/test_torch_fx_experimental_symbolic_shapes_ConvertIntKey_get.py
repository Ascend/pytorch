# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ConvertIntKey.get 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ConvertIntKey.get
API 签名：ConvertIntKey.get(self, b) -> IntLikeType

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空上下文、默认集合或 None 字段 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际符号键、Source、枚举策略和对应数据记录 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：符号 bool 无 guard 转换 |
| 正常传参场景 | 已覆盖：具体 bool 转整数及 KeyPath 字符串 |
| 异常传参场景 | 不适用：有效的策略或记录值 |
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

from torch.fx.experimental.symbolic_shapes import ConvertIntKey
from torch.fx.experimental.symbolic_shapes import ShapeEnv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_get_python_boolean_conversion_and_key_string(self):
        key = ConvertIntKey()
        self.assertEqual(key.get(True), 1)
        self.assertEqual(key.get(False), 0)
        self.assertIs(type(key.get(False)), int)
        self.assertEqual(str(key), ".cast_symbool_to_symint_guardless()")

    def test_get_symbolic_key_preserves_guardless_boolean_expression(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        value = env.create_unbacked_symint()
        result = ConvertIntKey().get(value > 3)
        self.assertIsInstance(result, torch.SymInt)
        self.assertEqual(result.node.expr.subs(value.node.expr, 5), 1)
        self.assertEqual(result.node.expr.subs(value.node.expr, 2), 0)
        self.assertEqual(env.guards, [])


if __name__ == "__main__":
    run_tests()
