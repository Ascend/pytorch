# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.asgd.ASGD.zero_grad 接口功能正确性
API 名称：torch.optim.asgd.ASGD.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数被清空；无梯度（None）参数不受影响 | 已覆盖（test_zero_grad_set_to_none / test_zero_grad_none_grad_untouched） |
| 枚举选项 | set_to_none=True / False 四分支：显式 True、显式 False、默认调用、truthy/falsy 归一 | 已覆盖（zero_grad_explicit_true/
zero_grad_set_to_false/zero_grad_default_equals_explicit_true/zero_grad_invalid） |
| 参数类型 | set_to_none 布尔；truthy/falsy 非布尔值按真假归一（无类型校验） | 已覆盖（test_zero_grad_invalid） |
| 传参与不传参 | 默认调用（等价显式 True）与显式 set_to_none=True/False | 已覆盖（zero_grad_default_equals_explicit_true/
zero_grad_explicit_true/zero_grad_set_to_false） |
| 等价类/边界值 | set_to_none=False 为 in-place 置 0：assertIs 引用保持、值全 0、device/dtype/shape 保持 | 已覆盖（test_zero_grad_set_to_false） |
| 正常传参场景 | 真实 backward 产生的梯度被清空；多参数组各自全部清空 | 已覆盖（test_zero_grad_after_real_backward / test_zero_grad_multiple_param_groups） |
| 异常传参场景 | zero_grad 基类无类型校验不抛异常：truthy/falsy 分别按 True/False 处理 | 已覆盖（test_zero_grad_invalid） |
| 混合设备类型 | 多参数组同设备清梯度；真实 backward 在当前设备（NPU/CUDA）上执行后清空 | 已覆盖（zero_grad_multiple_param_groups/
zero_grad_after_real_backward 在 self.device 上构造与反向） |

未覆盖项及原因：
- 跨设备混合参数组（CPU+NPU 参数同属一个优化器）：优化器允许混设备参数组，但该场景属
  Optimizer 构造契约而非 zero_grad 本身语义，且单卡环境下收益有限，不在本文件覆盖。
- 本接口为纯状态清空行为，无浮点数值输出，故无 CPU Golden 对照；置零结果以
  torch.testing.assert_close 对 zeros 断言。
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
class TestASGDzero_grad(TestCase):
    """Comprehensive cases for torch.optim.asgd.ASGD.zero_grad (NPU/CUDA portable).

    ASGD inherits zero_grad from Optimizer, but this file still covers the full
    four-branch contract independently: explicit set_to_none=True (grad becomes
    None), explicit set_to_none=False (in-place zeroing with reference
    identity kept), the default call (equivalent to explicit True), and the
    truthy/falsy normalization (no type validation). Extra scenarios: clearing
    grads produced by a real backward pass, clearing across multiple param
    groups, and None grads staying untouched.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.05):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.ASGD([p], lr=lr), p

    # ------------------------------------------------------------------
    # Branch 1/3: set_to_none=True (explicit) and the default call
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none(self):
        """default set_to_none=True leaves grad None."""
        opt, p = self._make_optimizer()
        p.grad = torch.randn_like(p)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_zero_grad_explicit_true(self):
        """Explicit set_to_none=True sets the grad to None."""
        opt, p = self._make_optimizer()
        p.grad = torch.randn_like(p)
        self.assertIsNotNone(p.grad, "precondition: a grad must exist")
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "explicit set_to_none=True must set grad to None")

    def test_zero_grad_default_equals_explicit_true(self):
        """The default call behaves exactly like explicit set_to_none=True."""
        opt_d, p_d = self._make_optimizer()
        opt_e, p_e = self._make_optimizer()
        # Identical grads on both optimizers.
        p_d.grad = torch.ones_like(p_d)
        p_e.grad = torch.ones_like(p_e)
        opt_d.zero_grad()                 # default
        opt_e.zero_grad(set_to_none=True)  # explicit
        # Equivalent outcome: both grads are gone (None), not zeroed tensors.
        self.assertIsNone(p_d.grad, "default call must behave as set_to_none=True")
        self.assertIsNone(p_e.grad, "explicit True must set grad to None")

    # ------------------------------------------------------------------
    # Branch 2/4: set_to_none=False (in-place zeroing, reference identity)
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeros the tensor instead of None."""
        opt, p = self._make_optimizer()
        p.grad = torch.ones_like(p)
        grad_ref = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad)
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))
        # The zeroing is in-place: the SAME tensor object is kept (assertIs),
        # not replaced by a fresh zeros tensor.
        self.assertIs(p.grad, grad_ref,
                      "set_to_none=False must zero in place, keeping the reference")
        # Metadata is untouched by the in-place zeroing.
        self.assertEqual(p.grad.device.type, self.device.type)
        self.assertEqual(p.grad.dtype, p.dtype)
        self.assertEqual(tuple(p.grad.shape), tuple(p.shape))

    def test_zero_grad_invalid(self):
        """Truthy set_to_none behaves as True, falsy as False."""
        opt, p = self._make_optimizer()
        # zero_grad does not type-check set_to_none; any truthy value is
        # treated as True and zeroes to None, any falsy value as False.
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none="yes")
        self.assertIsNone(p.grad, "truthy set_to_none must be treated as True")
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none=0)
        self.assertIsNotNone(p.grad, "falsy set_to_none must be treated as False")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # Real-workflow scenario: backward then zero
    # ------------------------------------------------------------------

    def test_zero_grad_after_real_backward(self):
        """Grads produced by a real backward pass are cleared on this device."""
        x = torch.nn.Parameter(torch.randn(4, device=self.device), requires_grad=True)
        opt = torch.optim.ASGD([x], lr=0.05)
        ((x * 2) ** 2).sum().backward()
        self.assertIsNotNone(x.grad, "precondition: backward must produce a grad")
        self.assertNotEqual(float(x.grad.abs().sum()), 0.0)
        opt.zero_grad()
        self.assertIsNone(x.grad, "zero_grad must clear the backward-produced grad")
        # And the set_to_none=False path zeroes the real grad in place too.
        ((x * 2) ** 2).sum().backward()
        ref = x.grad
        opt.zero_grad(set_to_none=False)
        torch.testing.assert_close(x.grad, torch.zeros_like(x.grad))
        self.assertIs(x.grad, ref)

    # ------------------------------------------------------------------
    # Multiple parameter groups
    # ------------------------------------------------------------------

    def test_zero_grad_multiple_param_groups(self):
        """zero_grad clears the grads of every parameter group."""
        p1 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.randn(5, device=self.device), requires_grad=True)
        p3 = torch.nn.Parameter(torch.randn(2, device=self.device), requires_grad=True)
        opt = torch.optim.ASGD([{"params": [p1], "lr": 0.05},
                                {"params": [p2, p3], "lr": 0.06}])
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        p3.grad = torch.ones_like(p3)
        opt.zero_grad()
        self.assertIsNone(p1.grad, "group-1 grad must be cleared")
        self.assertIsNone(p2.grad, "group-2 first grad must be cleared")
        self.assertIsNone(p3.grad, "group-2 second grad must be cleared")
        # The set_to_none=False path also covers every group in place.
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        p3.grad = torch.ones_like(p3)
        opt.zero_grad(set_to_none=False)
        for p in (p1, p2, p3):
            torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # None grads stay untouched
    # ------------------------------------------------------------------

    def test_zero_grad_none_grad_untouched(self):
        """A param whose grad is already None stays None without any error."""
        opt, p = self._make_optimizer()
        p.grad = None
        # Both branches must tolerate a None grad and leave it as None.
        opt.zero_grad()
        self.assertIsNone(p.grad)
        opt.zero_grad(set_to_none=False)
        self.assertIsNone(p.grad,
                          "set_to_none=False must not materialize a zeros grad")


if __name__ == "__main__":
    run_tests()
