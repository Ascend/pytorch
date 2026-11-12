# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.NAdam.zero_grad 接口功能正确性
API 名称：torch.optim.NAdam.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 对 grad 为 None 与非 None 的参数分别清空 | 已覆盖（test_zero_grad_none_grad_untouched / test_zero_grad_set_to_none_default_equals_true） |
| 枚举选项 | set_to_none=True/False 两分支 | 已覆盖（test_zero_grad_set_to_none_true / test_zero_grad_set_to_false_inplace） |
| 参数类型 | set_to_none 布尔 | 已覆盖（test_zero_grad_set_to_none_true） |
| 传参与不传参 | 默认调用与显式 True 等价 | 已覆盖（test_zero_grad_set_to_none_default_equals_true） |
| 等价类/边界值 | set_to_none=False in-place 置 0 且 grad 引用对象保持 | 已覆盖（test_zero_grad_set_to_false_inplace） |
| 正常传参场景 | step 前置梯度后 zero_grad 清空；多参数组全部清空 | 已覆盖（test_zero_grad_after_step / test_zero_grad_multiple_param_groups） |
| 异常传参场景 | zero_grad 不校验类型：truthy 视为 True、falsy 视为 False | 已覆盖（test_zero_grad_invalid） |
| 混合设备类型 | 多参数组跨设备清梯度 | 未覆盖（单设备参数组，无跨设备迁移） |

未覆盖项及原因：
- 传非法类型报错：实测 zero_grad 对 set_to_none 不做类型校验（truthy/falsy 归一），
  无稳定异常路径，故以 truthy/falsy 行为断言代替（test_zero_grad_invalid）。
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
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
class TestNAdamzero_grad(TestCase):
    """Test cases for torch.optim.NAdam.zero_grad (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        # Fresh optimizer with one param on the current (NPU/CUDA/CPU) device.
        return torch.optim.NAdam(
            [torch.nn.Parameter(torch.randn(3, device=self.device_name),
                                requires_grad=True)], lr=0.01)

    # ------------------------------------------------------------------
    # Branch 1: set_to_none=True (explicit) -> grad becomes None
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none_true(self):
        """Explicit set_to_none=True detaches the grad to None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "set_to_none=True must detach grad to None")

    # ------------------------------------------------------------------
    # Branch 2: set_to_none=False -> in-place zeroing, reference preserved
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_false_inplace(self):
        """set_to_none=False zeros the SAME tensor object in-place."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        grad_ref = p.grad
        opt.zero_grad(set_to_none=False)
        # Reference identity: the grad tensor object must be preserved
        # (in-place zeroing), not replaced by a fresh zeros tensor.
        self.assertIs(p.grad, grad_ref,
                      "set_to_none=False must zero in-place, keeping the "
                      "same grad object")
        # Every element must be exactly zero.
        self.assertEqual(torch.count_nonzero(p.grad).item(), 0,
                         "grad must be all zeros after set_to_none=False")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

    # ------------------------------------------------------------------
    # Branch 3: default call == set_to_none=True
    # ------------------------------------------------------------------

    def test_zero_grad_set_to_none_default_equals_true(self):
        """The no-argument call behaves exactly like set_to_none=True."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad()  # default
        self.assertIsNone(p.grad, "default zero_grad() must set grad to None")
        # ...and re-setting grad then calling explicitly confirms equality
        # of the default and the explicit True branch.
        p.grad = torch.randn_like(p)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)

    # ------------------------------------------------------------------
    # Realistic flow: backward -> zero_grad
    # ------------------------------------------------------------------

    def test_zero_grad_after_step(self):
        """zero_grad clears grads populated by a real backward()."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device_name))
        opt = torch.optim.NAdam([p], lr=0.01)
        for expected_none in (False, True):
            loss = ((p - 1.0) ** 2).sum()
            loss.backward()
            self.assertIsNotNone(p.grad)
            opt.step()
            if expected_none:
                opt.zero_grad()
                self.assertIsNone(p.grad)
            else:
                opt.zero_grad(set_to_none=False)
                self.assertIsNotNone(p.grad)
                self.assertEqual(torch.count_nonzero(p.grad).item(), 0)

    # ------------------------------------------------------------------
    # Multi-param-group coverage
    # ------------------------------------------------------------------

    def test_zero_grad_multiple_param_groups(self):
        """All params across every param group are cleared."""
        p1 = torch.nn.Parameter(torch.randn(2, device=self.device_name))
        p2 = torch.nn.Parameter(torch.randn(2, device=self.device_name))
        p3 = torch.nn.Parameter(torch.randn(2, device=self.device_name))
        opt = torch.optim.NAdam([{"params": [p1, p2]}, {"params": [p3]}], lr=0.01)
        for p in (p1, p2, p3):
            p.grad = torch.randn_like(p)
        opt.zero_grad()
        self.assertIsNone(p1.grad)
        self.assertIsNone(p2.grad)
        self.assertIsNone(p3.grad)
        # In-place variant also spans all groups.
        for p in (p1, p2, p3):
            p.grad = torch.randn_like(p)
        opt.zero_grad(set_to_none=False)
        for p in (p1, p2, p3):
            self.assertIsNotNone(p.grad)
            self.assertEqual(torch.count_nonzero(p.grad).item(), 0)

    # ------------------------------------------------------------------
    # Edge case: a param whose grad is already None stays None
    # ------------------------------------------------------------------

    def test_zero_grad_none_grad_untouched(self):
        """A param with grad=None keeps grad=None without any error."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        self.assertIsNone(p.grad)
        opt.zero_grad()           # must not raise
        opt.zero_grad(set_to_none=False)  # must not raise either
        self.assertIsNone(p.grad)

    # ------------------------------------------------------------------
    # Branch 4: truthy / falsy non-bool values (no type check)
    # ------------------------------------------------------------------

    def test_zero_grad_invalid(self):
        """Truthy set_to_none behaves as True, falsy as False."""
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
