# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.radam.RAdam.zero_grad 接口功能正确性
API 名称：torch.optim.radam.RAdam.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数被清空；从未有梯度的参数不受影响 | 已覆盖（test_zero_grad_set_to_none / test_zero_grad_param_without_grad_untouched） |
| 枚举选项 | set_to_none=True/False 两分支 | 已覆盖（test_zero_grad_set_to_none / test_zero_grad_set_to_false / test_zero_grad_explicit_true） |
| 参数类型 | set_to_none 布尔；truthy/falsy 强制转换 | 已覆盖（test_zero_grad_invalid_type_truthy_falsy） |
| 传参与不传参 | 默认调用（等价 set_to_none=True）与显式传参 | 已覆盖（zero_grad_set_to_none/zero_grad_explicit_true/zero_grad_set_to_false） |
| 等价类/边界值 | set_to_none=False 时 in-place 置 0 且梯度张量引用保持不变 | 已覆盖（test_zero_grad_set_to_false_keeps_reference） |
| 正常传参场景 | step 后 zero_grad 清空梯度、多参数组全部清空 | 已覆盖（test_zero_grad_after_backward / test_zero_grad_multiple_param_groups） |
| 异常传参场景 | zero_grad 不校验类型：truthy 视为 True、falsy 视为 False | 已覆盖（test_zero_grad_invalid_type_truthy_falsy） |
| 混合设备类型 | 参数张量位于当前设备（NPU/CUDA 可移植） | 已覆盖（各用例均使用 self.device 创建参数） |

未覆盖项及原因：
- 无。本接口为纯状态清空操作，无数值对照语义；浮点断言仅用于校验置零结果，
  属于设备无关的确定性比较（zeros 对照）。

实测核实记录（python -c，torch 2.11）：
- 默认 set_to_none=True：p.grad 变 None。
- set_to_none=False：p.grad 与原张量为同一对象（引用保持），且数值全 0。
- truthy（"yes"）按 True 处理、falsy（0）按 False 处理，不抛 TypeError。
- 从未赋梯度的参数在 zero_grad 后仍为 None，无任何副作用。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 条件导入 torch_npu，并用 torch.npu.is_available() 探测可用 NPU。
DEVICE_NAME = None
try:
    import torch_npu  # noqa: F401
    if torch.npu.is_available():
        DEVICE_NAME = 'npu'
except (ImportError, AttributeError):
    pass
if DEVICE_NAME is None:
    DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRAdamzero_grad(TestCase):
    """Test cases for torch.optim.radam.RAdam.zero_grad (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        return torch.optim.RAdam(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)], lr=0.01)

    # ------------------------------------------------------------------
    # Branch 1: set_to_none=True (also the default)
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none(self):
        """Default call (no argument) sets grad to None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_zero_grad_explicit_true(self):
        """Explicit set_to_none=True behaves the same as the default."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)

    # ------------------------------------------------------------------
    # Branch 2: set_to_none=False (in-place zero + reference kept)
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeros the tensor instead of setting None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad)
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    def test_zero_grad_set_to_false_keeps_reference(self):
        """set_to_none=False zeroes in-place: the grad Tensor object is preserved."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        g = torch.randn_like(p)
        p.grad = g
        opt.zero_grad(set_to_none=False)
        self.assertIs(p.grad, g, "set_to_none=False must zero the SAME tensor, not replace it")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # Branch 3 (gap-fill): default vs explicit equivalence, real backward flow
    # ------------------------------------------------------------------

    def test_zero_grad_after_backward(self):
        """Gradients populated by a real backward() are cleared by the default call."""
        x = torch.nn.Parameter(torch.ones(2, device=self.device))
        opt = torch.optim.RAdam([x], lr=0.01)
        loss = ((x - 3) ** 2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        opt.zero_grad()
        self.assertIsNone(x.grad)
        # After zeroing, a fresh backward must re-populate the grad normally.
        ((x - 3) ** 2).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_zero_grad_multiple_param_groups(self):
        """zero_grad clears grads across ALL param groups."""
        p1 = torch.nn.Parameter(torch.randn(2, device=self.device))
        p2 = torch.nn.Parameter(torch.randn(2, device=self.device))
        opt = torch.optim.RAdam(
            [{"params": [p1], "lr": 0.01}, {"params": [p2], "lr": 0.02}])
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.full_like(p2, 2.0)
        opt.zero_grad()
        self.assertIsNone(p1.grad)
        self.assertIsNone(p2.grad)

    def test_zero_grad_param_without_grad_untouched(self):
        """A param that never received a grad stays None (no error, no tensor created)."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        opt.zero_grad()
        self.assertIsNone(p.grad)
        opt.zero_grad(set_to_none=False)
        # The no-grad param must not be given a zero tensor by set_to_none=False.
        self.assertIsNone(p.grad)

    # ------------------------------------------------------------------
    # Branch 4: invalid types are not rejected (truthy/falsy coercion)
    # ------------------------------------------------------------------

    def test_zero_grad_invalid_type_truthy_falsy(self):
        """Truthy set_to_none behaves as True, falsy as False; no TypeError raised."""
        opt = self._make_optimizer()
        # zero_grad does not type-check set_to_none; any truthy value is
        # treated as True and zeroes to None, any falsy value as False.
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none="yes")
        self.assertIsNone(p.grad, "truthy set_to_none must be treated as True")
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none=0)
        self.assertIsNotNone(p.grad, "falsy set_to_none must be treated as False")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))


if __name__ == "__main__":
    run_tests()
