# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adadelta.Adadelta.zero_grad 接口功能正确性
API 名称：torch.optim.adadelta.Adadelta.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数被清空；无梯度（grad=None）参数不受影响且不物化 | 已覆盖（test_zero_grad_set_to_none/test_zero_grad_none_grad_untouched） |
| 枚举选项 | set_to_none=True/False 两分支 + 默认（=显式 True 等价） |
已覆盖（zero_grad_set_to_none/zero_grad_false_in_place_reference_kept/
zero_grad_default_equals_explicit_true/zero_grad_explicit_true） |
| 参数类型 | set_to_none 布尔；无类型校验：truthy 归一为 True、falsy 归一为 False | 已覆盖（test_zero_grad_invalid） |
| 传参与不传参 | 默认调用与显式传参 | 已覆盖（test_zero_grad_set_to_none/test_zero_grad_explicit_true/test_zero_grad_default_equals_explicit_true） |
| 等价类/边界值 | set_to_none=False 为 in-place 置 0 且 assertIs 引用保持；清空后可再次 backward 且不累积旧梯度 |
已覆盖（zero_grad_false_in_place_reference_kept/
zero_grad_real_backward_rebackward/zero_grad_false_no_stale_accumulation） |
| 正常传参场景 | 真实 backward 产生梯度后清空可再 backward；多参数组（含从未有梯度的参数）全部清空 | 已覆盖（zero_grad_real_backward_rebackward/
zero_grad_multiple_param_groups） |
| 异常传参场景 | zero_grad 无类型校验：truthy/falsy 均不抛异常，按布尔语义归一 | 已覆盖（test_zero_grad_invalid） |
| 混合设备类型 | 梯度张量位于当前设备（NPU/CUDA 可移植）被原位清零 | 已覆盖（test_zero_grad_false_in_place_reference_kept 在 self.device 上创建） |

未覆盖项及原因：
- 无。
- 本接口为框架行为，无浮点数值设备对照语义（仅置 None/置 0 的结构与引用断言），涉及清零后数值用 torch.testing.assert_close 校验。

实测说明：
- zero_grad 对 set_to_none 无类型校验：set_to_none="yes" 等价 True（置 None），set_to_none=0 等价 False（in-place 置 0）。
- set_to_none=False 时梯度张量引用保持（assertIs 成立），即原地置 0 而非新建张量。
- grad=None 的参数在 set_to_none=False 下保持 None，不会被物化为 0 张量。
- 置 0 后再次 backward 得到 fresh 梯度（2.0 而非 4.0），证明旧梯度确实被清零。
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
class TestAdadeltazero_grad(TestCase):
    """Comprehensive cases for torch.optim.adadelta.Adadelta.zero_grad (NPU/CUDA portable).

    Covers the four set_to_none branches (default, explicit True, explicit
    False, truthy/falsy normalization), the in-place zeroing contract with
    reference identity (assertIs), grad-less params staying None, real
    backward -> zero_grad -> backward again flows (no stale accumulation)
    and multi-param-group clearing including a param that never had a grad.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # Branch 1/3: set_to_none=True (default and explicit) -> grad becomes None
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none(self):
        """default set_to_none=True leaves grad None."""
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_zero_grad_explicit_true(self):
        """Explicit set_to_none=True sets the grad to None."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.01)
        p.grad = torch.randn_like(p)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "explicit set_to_none=True must set grad to None")

    def test_zero_grad_default_equals_explicit_true(self):
        """The default call is equivalent to the explicit set_to_none=True."""
        p_def = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        p_exp = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt_def = torch.optim.Adadelta([p_def], lr=0.01)
        opt_exp = torch.optim.Adadelta([p_exp], lr=0.01)
        p_def.grad = torch.ones_like(p_def)
        p_exp.grad = torch.ones_like(p_exp)
        opt_def.zero_grad()              # default
        opt_exp.zero_grad(set_to_none=True)  # explicit
        self.assertIsNone(p_def.grad, "default call must clear to None")
        self.assertIsNone(p_exp.grad, "explicit True must clear to None")
        self.assertEqual(p_def.grad, p_exp.grad,
                         "default and explicit True must produce identical results")

    # ------------------------------------------------------------------
    # Branch 2/3: set_to_none=False -> in-place zeroing, reference kept
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeros the tensor instead of None."""
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad)
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    def test_zero_grad_false_in_place_reference_kept(self):
        """set_to_none=False zeroes in place: the grad object identity is kept."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.01)
        p.grad = torch.ones_like(p)
        grad_ref = p.grad
        before_dtype, before_shape = p.grad.dtype, p.grad.shape
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad, "set_to_none=False must keep a Tensor grad")
        # In-place contract: the very same grad object is zeroed, not replaced.
        self.assertIs(p.grad, grad_ref,
                      "set_to_none=False must zero the existing grad in place")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad),
                                   msg="the kept grad must be all zeros")
        self.assertEqual(p.grad.dtype, before_dtype, "dtype must be preserved")
        self.assertEqual(p.grad.shape, before_shape, "shape must be preserved")

    # ------------------------------------------------------------------
    # Branch 4/3: no type checking, truthy/falsy normalization
    # ------------------------------------------------------------------

    def test_zero_grad_invalid(self):
        """Truthy set_to_none behaves as True, falsy as False."""
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
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

    # ------------------------------------------------------------------
    # Grad-less params are untouched (never materialized into zeros)
    # ------------------------------------------------------------------

    def test_zero_grad_none_grad_untouched(self):
        """A param whose grad is already None stays None under both modes."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.01)
        opt.zero_grad(set_to_none=False)
        self.assertIsNone(p.grad,
                          "set_to_none=False must not materialize a grad for a None-grad param")
        opt.zero_grad()
        self.assertIsNone(p.grad)

    # ------------------------------------------------------------------
    # Real autograd flows: zero_grad between backwards
    # ------------------------------------------------------------------

    def test_zero_grad_real_backward_rebackward(self):
        """After a real backward, zero_grad() allows a fresh backward again."""
        x = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([x], lr=0.01)
        ((x * 2) ** 2).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 8.0),
                                   msg="d(4x^2)/dx = 8x = 8 for x=1")
        opt.zero_grad()
        self.assertIsNone(x.grad, "zero_grad() must clear the accumulated grad")
        # A second backward recomputes a fresh grad (not accumulated).
        ((x * 2) ** 2).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 8.0),
                                   msg="backward after zero_grad must produce a fresh grad")

    def test_zero_grad_false_no_stale_accumulation(self):
        """set_to_none=False really zeroes: the next backward does not accumulate stale grads."""
        x = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([x], lr=0.01)
        (x * 2).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2.0),
                                   msg="grad after the first backward must be 2")
        opt.zero_grad(set_to_none=False)
        torch.testing.assert_close(x.grad, torch.zeros_like(x),
                                   msg="grad must be zeroed in place")
        # If zeroing had failed, this backward would accumulate to 4.0.
        (x * 2).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2.0),
                                   msg="grad must restart from zero, not accumulate to 4")

    # ------------------------------------------------------------------
    # Multiple param groups
    # ------------------------------------------------------------------

    def test_zero_grad_multiple_param_groups(self):
        """All groups are cleared; a param that never had a grad stays None."""
        p1 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        p3 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([
            {"params": [p1], "lr": 0.1},
            {"params": [p2, p3], "lr": 0.2},
        ])
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        # Default call clears every group to None.
        opt.zero_grad()
        self.assertIsNone(p1.grad, "group-1 param grad must be None")
        self.assertIsNone(p2.grad, "group-2 param grad must be None")
        self.assertIsNone(p3.grad, "never-had-grad param must stay None")
        # set_to_none=False zeroes both groups in place, reference kept.
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        g2_ref = p2.grad
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p1.grad)
        self.assertIsNotNone(p2.grad)
        self.assertIs(p2.grad, g2_ref, "in-place zeroing must keep the grad object")
        torch.testing.assert_close(p1.grad, torch.zeros_like(p1.grad))
        torch.testing.assert_close(p2.grad, torch.zeros_like(p2.grad))
        self.assertIsNone(p3.grad,
                          "set_to_none=False must not materialize grads for grad-less params")


if __name__ == "__main__":
    run_tests()
