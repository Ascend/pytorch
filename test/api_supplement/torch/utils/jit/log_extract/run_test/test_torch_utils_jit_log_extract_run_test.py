# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.jit.log_extract.run_test 接口功能正确性
API 名称：torch.utils.jit.log_extract.run_test
API 签名：torch.utils.jit.log_extract.run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 无 Tensor 输入与有效 Tensor 输入列表 | 已覆盖（non_tensor_inputs_raises、custom_runs）    |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | Tensor 列表、非 Tensor 列表与整型参数| 已覆盖（custom_runs、non_tensor_inputs_raises）    |
| 传参与不传参     | 可选关键字参数省略/显式传入与缺参    | 已覆盖（default_runs、custom_runs、                |
|                  |                                      | missing_args_raises）                              |
| 等价类/边界值    | 仅关键字参数位置传参拒绝与小轮次计数 | 已覆盖（keyword_only_args_reject_positional、      |
|                  |                                      | custom_runs）                                      |
| 正常传参场景     | 默认与自定义 warmup/test 轮次计时    | 已覆盖（default_runs、custom_runs）                |
| 异常传参场景     | 位置传参、无 Tensor、非法 IR 与缺参  | 已覆盖（keyword_only_args_reject_positional、      |
|                  |                                      | non_tensor_inputs_raises、invalid_ir_raises、      |
|                  |                                      | missing_args_raises）                              |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（JIT IR 显式绑定单设备，不支持混合设备）    |

未覆盖项及原因：
- 枚举选项：该接口参数为 IR 字符串、输入列表与整型迭代次数，无枚举类参数。
- 混合设备类型：TorchScript IR 中各节点与输入张量设备在图内静态声明，本接口用于同设备图执行计时，不涉及 NPU/CPU 混合设备场景。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import math
import os
import unittest

import torch
from torch.utils.jit.log_extract import run_test

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


SIMPLE_ADD_IR = """
graph(%x : Float(4, 4, strides=[4, 1], device=cpu),
      %y : Float(4, 4, strides=[4, 1], device=cpu)):
  %1 : int = prim::Constant[value=1]()
  %z : Float(4, 4, strides=[4, 1], device=cpu) = aten::add(%x, %y, %1)
  return (%z)
"""

INT_ADD_IR = """
graph(%a : int, %b : int):
  %c : int = aten::add(%a, %b)
  return (%c)
"""


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRunTest(TestCase):
    """Test cases for torch.utils.jit.log_extract.run_test."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.inputs = [torch.ones(4, 4), torch.ones(4, 4)]

    def test_custom_runs(self):
        """run_test with small warmup_runs and test_runs returns non-negative float."""
        elapsed_ms = run_test(SIMPLE_ADD_IR, self.inputs, warmup_runs=1, test_runs=2)
        self.assertIsInstance(elapsed_ms, float)
        self.assertFalse(math.isnan(elapsed_ms))
        self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_default_runs(self):
        """run_test with default warmup_runs/test_runs returns non-negative float."""
        elapsed_ms = run_test(SIMPLE_ADD_IR, self.inputs)
        self.assertIsInstance(elapsed_ms, float)
        self.assertFalse(math.isnan(elapsed_ms))
        self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_keyword_only_args_reject_positional(self):
        """Passing warmup_runs or test_runs positionally raises TypeError."""
        with self.assertRaises(TypeError):
            run_test(SIMPLE_ADD_IR, self.inputs, 1, 2)

    def test_non_tensor_inputs_raises(self):
        """Inputs containing no torch.Tensor raise AssertionError."""
        with self.assertRaises(AssertionError):
            run_test(INT_ADD_IR, [1, 2], warmup_runs=1, test_runs=1)

    def test_invalid_ir_raises(self):
        """Malformed TorchScript IR raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            run_test("invalid graph ir", self.inputs, warmup_runs=1, test_runs=1)

    def test_missing_args_raises(self):
        """Omitting required ir or inputs argument raises TypeError."""
        with self.assertRaises(TypeError):
            run_test()
        with self.assertRaises(TypeError):
            run_test(SIMPLE_ADD_IR)


if __name__ == "__main__":
    run_tests()
