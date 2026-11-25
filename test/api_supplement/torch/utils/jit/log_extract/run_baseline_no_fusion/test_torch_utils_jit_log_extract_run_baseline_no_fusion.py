# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.jit.log_extract.run_baseline_no_fusion 接口功能正确性
API 名称：torch.utils.jit.log_extract.run_baseline_no_fusion
API 签名：torch.utils.jit.log_extract.run_baseline_no_fusion(ir, inputs) -> float

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空 inputs 列表与有效 Tensor 输入列表 | 已覆盖（empty_inputs_raises、basic_cpu_ir）        |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | IR 字符串与 Tensor 列表参数          | 已覆盖（basic_cpu_ir、disables_optimize_and_restores）|
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（basic_cpu_ir、missing_args_raises）        |
| 等价类/边界值    | 禁用图优化并恢复状态、非法 IR        | 已覆盖（disables_optimize_and_restores、           |
|                  |                                      | invalid_ir_raises）                                |
| 正常传参场景     | CPU IR 基准耗时返回非负浮点数        | 已覆盖（basic_cpu_ir、disables_optimize_and_restores）|
| 异常传参场景     | 非法 IR、空 inputs 与缺参异常        | 已覆盖（invalid_ir_raises、empty_inputs_raises、   |
|                  |                                      | missing_args_raises）                              |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（JIT IR 显式绑定单设备，不支持混合设备）    |

未覆盖项及原因：
- 枚举选项：该接口仅接收 ir 字符串与 inputs 列表，无枚举类参数。
- 混合设备类型：TorchScript IR 中各节点与输入张量设备在图内静态声明，本接口用于同设备基准计时，不涉及 NPU/CPU 混合设备场景。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import math
import os
import unittest
from unittest.mock import patch

import torch
from torch.utils.jit.log_extract import run_baseline_no_fusion

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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRunBaselineNoFusion(TestCase):
    """Test cases for torch.utils.jit.log_extract.run_baseline_no_fusion."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.inputs = [torch.ones(4, 4), torch.ones(4, 4)]

    def test_basic_cpu_ir(self):
        """Running baseline no-fusion on a valid CPU IR returns non-negative float."""
        elapsed_ms = run_baseline_no_fusion(SIMPLE_ADD_IR, self.inputs)
        self.assertIsInstance(elapsed_ms, float)
        self.assertFalse(math.isnan(elapsed_ms))
        self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_disables_optimize_and_restores(self):
        """no_fuser disables graph executor optimization during run_test and restores it."""
        orig_opt = torch._C._get_graph_executor_optimize()
        observed = []

        def fake_run_test(ir, inputs):
            observed.append((ir, inputs, torch._C._get_graph_executor_optimize()))
            return 1.25

        with patch("torch.utils.jit.log_extract.run_test", side_effect=fake_run_test):
            ret = run_baseline_no_fusion(SIMPLE_ADD_IR, self.inputs)

        self.assertEqual(ret, 1.25)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], SIMPLE_ADD_IR)
        self.assertIs(observed[0][1], self.inputs)
        self.assertFalse(observed[0][2])
        self.assertEqual(torch._C._get_graph_executor_optimize(), orig_opt)

    def test_invalid_ir_raises(self):
        """Malformed TorchScript IR string raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            run_baseline_no_fusion("not valid graph ir", self.inputs)

    def test_empty_inputs_raises(self):
        """Passing empty inputs list for a 2-input graph raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            run_baseline_no_fusion(SIMPLE_ADD_IR, [])

    def test_missing_args_raises(self):
        """Omitting required ir or inputs argument raises TypeError."""
        with self.assertRaises(TypeError):
            run_baseline_no_fusion()
        with self.assertRaises(TypeError):
            run_baseline_no_fusion(SIMPLE_ADD_IR)


if __name__ == "__main__":
    run_tests()
