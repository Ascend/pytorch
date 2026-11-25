# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.checkpoint.SelectiveCheckpointContext 接口功能正确性
API 名称：torch.utils.checkpoint.SelectiveCheckpointContext
API 签名：torch.utils.checkpoint.SelectiveCheckpointContext(*, is_recompute, op_output=None)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | op_output 默认 None 与显式非空对象   | 已覆盖（is_recompute_false_default_op_output、explicit_op_output） |
| 枚举选项         | is_recompute 布尔值 True / False     | 已覆盖（is_recompute_false_default_op_output、is_recompute_true） |
| 参数类型         | dict 对象与目标设备 Tensor 作为输出  | 已覆盖（explicit_op_output、tensor_op_output）     |
| 传参与不传参     | op_output 省略与显式传入、缺必选参数 | 已覆盖（is_recompute_false_default_op_output、missing_is_recompute_raises） |
| 等价类/边界值    | 仅关键字参数约束与对象引用身份保持   | 已覆盖（positional_arg_raises、explicit_op_output、tensor_op_output） |
| 正常传参场景     | 正常构造并读取 is_recompute/op_output | 已覆盖（is_recompute_false_default_op_output、tensor_op_output 等） |
| 异常传参场景     | 缺参、位置传参与未知关键字报 TypeError | 已覆盖（missing_is_recompute_raises、positional_arg_raises 等） |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：轻量级数据容器类，仅保存单个 op_output 引用，不涉及多 Tensor 设备交互。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.checkpoint import SelectiveCheckpointContext

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestSelectiveCheckpointContext(TestCase):
    """Test cases for torch.utils.checkpoint.SelectiveCheckpointContext."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_is_recompute_false_default_op_output(self):
        """Constructing with is_recompute=False sets op_output to None by default."""
        ctx = SelectiveCheckpointContext(is_recompute=False)
        self.assertFalse(ctx.is_recompute)
        self.assertIsNone(ctx.op_output)

    def test_is_recompute_true(self):
        """Constructing with is_recompute=True stores True accurately."""
        ctx = SelectiveCheckpointContext(is_recompute=True)
        self.assertTrue(ctx.is_recompute)
        self.assertIsNone(ctx.op_output)

    def test_explicit_op_output(self):
        """Explicit op_output preserves exact object identity."""
        sentinel = {"key": 123}
        ctx = SelectiveCheckpointContext(is_recompute=False, op_output=sentinel)
        self.assertFalse(ctx.is_recompute)
        self.assertIs(ctx.op_output, sentinel)

    def test_tensor_op_output(self):
        """Tensor passed as op_output on target device is stored by identity."""
        out = torch.ones(2, 3, device=self.device)
        ctx = SelectiveCheckpointContext(is_recompute=False, op_output=out)
        self.assertIs(ctx.op_output, out)
        self.assertEqual(ctx.op_output.shape, (2, 3))

    def test_missing_is_recompute_raises(self):
        """Omitting required keyword-only is_recompute raises TypeError."""
        with self.assertRaises(TypeError):
            SelectiveCheckpointContext()

    def test_positional_arg_raises(self):
        """Passing is_recompute positionally violates keyword-only signature."""
        with self.assertRaises(TypeError):
            SelectiveCheckpointContext(True)

    def test_unexpected_kwarg_raises(self):
        """Passing an unknown keyword argument raises TypeError."""
        with self.assertRaises(TypeError):
            SelectiveCheckpointContext(is_recompute=False, unknown=1)


if __name__ == "__main__":
    run_tests()
