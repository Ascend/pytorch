# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Adafactor.zero_grad 接口功能正确性
API 名称：torch.optim.Adafactor.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度和无梯度参数的 grad 处理 | 已覆盖（_set_to_none_true/_none_grad_stays_none） |
| 枚举选项 | set_to_none=True/False | 已覆盖（_set_to_none_true/_set_to_none_false_zeroes_in_place） |
| 参数类型 | set_to_none 布尔；实测不校验类型（truthy/falsy 生效） | 已覆盖（_untyped_argument） |
| 传参与不传参 | 默认（当前版本 True）与显式调用 | 已覆盖（_default/_set_to_none_true） |
| 等价类/边界值 | set_to_none=False 置零且引用保持；grad 已为 None 时保持 None | 已覆盖（_set_to_none_false_zeroes_in_place/_none_grad_stays_none） |
| 正常传参场景 | step 前置梯度后 zero_grad 清空；多参数组全部清空 | 已覆盖（_set_to_none_true/_multiple_params_and_groups） |
| 异常传参场景 | zero_grad 不校验类型：truthy 视为 True、falsy 视为 False | 已覆盖（_untyped_argument） |
| 混合设备类型 | 参数位于当前设备（NPU/CUDA 可移植）执行清梯度 | 已覆盖（_multiple_params_and_groups） |

未覆盖项及原因：
- 无（实测确认 zero_grad 对 set_to_none 不做类型校验，任何 truthy/falsy 值分别按
  True/False 生效，不抛异常，故「异常传参」维度的实测语义即 truthy/falsy 分支）。
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
class TestAdafactorzero_grad(TestCase):
    """Test cases for torch.optim.Adafactor.zero_grad (NPU/CUDA portable).

    Adafactor inherits zero_grad from Optimizer, but this file still covers
    the full four-branch contract independently: set_to_none=True / False
    (in-place zeroing with reference identity), the current default (True),
    and the untyped-argument semantics, plus edge cases (already-None grads,
    multiple param groups on the active device).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adafactor([p], lr=0.01)

    def test_zero_grad_set_to_none_true(self):
        """Explicit set_to_none=True detaches the gradient (grad becomes None)."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn(3, device=self.device)
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "set_to_none=True must leave grad as None")

    def test_zero_grad_set_to_none_false_zeroes_in_place(self):
        """set_to_none=False zeroes the tensor in place, keeping the reference."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones(3, device=self.device)
        grad_before = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad, "set_to_none=False must keep a materialized grad")
        # Reference identity: the same grad Tensor object is zeroed in place,
        # not replaced by a fresh allocation.
        self.assertIs(p.grad, grad_before,
                      "set_to_none=False must zero the existing grad object")
        torch.testing.assert_close(p.grad, torch.zeros(3, device=self.device))

    def test_zero_grad_default(self):
        """Default call (no argument) behaves as set_to_none=True in this version."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn(3, device=self.device)
        opt.zero_grad()
        self.assertIsNone(p.grad, "default zero_grad() must set grad to None")

    def test_zero_grad_untyped_argument(self):
        """zero_grad does not type-check set_to_none; truthy acts as True, falsy as False."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        # Verified behavior: no TypeError is raised for non-bool values; the
        # truthiness of the value selects the branch.
        p.grad = torch.ones(3, device=self.device)
        opt.zero_grad(set_to_none="yes")
        self.assertIsNone(p.grad, "truthy set_to_none must be treated as True")
        p.grad = torch.ones(3, device=self.device)
        opt.zero_grad(set_to_none=0)
        self.assertIsNotNone(p.grad, "falsy set_to_none must be treated as False")
        torch.testing.assert_close(p.grad, torch.zeros(3, device=self.device))

    def test_zero_grad_none_grad_stays_none(self):
        """A param whose grad is already None stays None even with set_to_none=False."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = None
        # The in-place branch skips params without grads; no error is raised.
        opt.zero_grad(set_to_none=False)
        self.assertIsNone(p.grad)

    def test_zero_grad_multiple_params_and_groups(self):
        """All params across multiple groups are cleared in one call."""
        p1 = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.randn(4, device=self.device), requires_grad=True)
        opt = torch.optim.Adafactor([{"params": [p1], "lr": 0.01},
                                     {"params": [p2], "lr": 0.02}])
        p1.grad = torch.randn(3, device=self.device)
        p2.grad = torch.randn(4, device=self.device)
        opt.zero_grad()
        self.assertIsNone(p1.grad, "group-1 grad must be cleared")
        self.assertIsNone(p2.grad, "group-2 grad must be cleared")


if __name__ == "__main__":
    run_tests()
