# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.set_original_aten_op 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.set_original_aten_op
API 签名：set_original_aten_op(func) -> Generator[None, None, None]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：异常恢复及未 preserve 时无操作 |
| 正常传参场景 | 已覆盖：ATen 原始算子上下文、嵌套外层优先 |
| 异常传参场景 | 已覆盖：受控异常后 ContextVar 恢复 |
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

from torch.fx.experimental import proxy_tensor as pt
from torch.fx import traceback as fx_traceback


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_set_original_aten_op_outer_operation_wins_and_context_state_is_restored(self):
        original = pt.ORIGINAL_ATEN.get()
        token = pt.ORIGINAL_ATEN.set(None)

        def restore():
            pt.ORIGINAL_ATEN.reset(token)
            self.assertIs(pt.ORIGINAL_ATEN.get(), original)

        self.addCleanup(restore)
        with fx_traceback.preserve_node_meta():
            with pt.set_original_aten_op(torch.ops.aten.add.Tensor):
                self.assertIs(pt.ORIGINAL_ATEN.get(), torch.ops.aten.add.Tensor)
                self.assertIs(fx_traceback.get_current_meta()["original_aten"], torch.ops.aten.add.Tensor)
                with pt.set_original_aten_op(torch.ops.aten.neg.default):
                    self.assertIs(pt.ORIGINAL_ATEN.get(), torch.ops.aten.add.Tensor)
            self.assertIsNone(pt.ORIGINAL_ATEN.get())
            self.assertIsNone(fx_traceback.get_current_meta()["original_aten"])

    def test_set_original_aten_op_exception_exit_restores_context_variable_and_no_preserve_is_noop(self):
        token = pt.ORIGINAL_ATEN.set(None)
        self.addCleanup(pt.ORIGINAL_ATEN.reset, token)
        with fx_traceback.preserve_node_meta():
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                with pt.set_original_aten_op(torch.ops.aten.neg.default):
                    raise RuntimeError("controlled")
            self.assertIsNone(pt.ORIGINAL_ATEN.get())
            self.assertIsNone(fx_traceback.get_current_meta()["original_aten"])
        with pt.set_original_aten_op(torch.ops.aten.add.Tensor):
            self.assertIsNone(pt.ORIGINAL_ATEN.get())


if __name__ == "__main__":
    run_tests()
