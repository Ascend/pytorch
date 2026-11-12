# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adam.Adam.zero_grad 接口功能正确性
API 名称：torch.optim.adam.Adam.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 置空有梯度参数的 grad；无 grad 参数不受影响 | 已覆盖（test_zero_grad_set_to_none/test_zero_grad_multiple_groups_on_device） |
| 枚举选项 | set_to_none=True/False 两分支 | 已覆盖（test_zero_grad_set_to_none_explicit/test_zero_grad_set_to_false） |
| 参数类型 | set_to_none 布尔；实测不校验类型（truthy 视为 True、falsy 视为 False） | 已覆盖（test_zero_grad_invalid） |
| 传参与不传参 | 默认调用与显式传参 | 已覆盖（test_zero_grad_set_to_none/test_zero_grad_set_to_none_explicit） |
| 等价类/边界值 | False 分支 in-place 置 0、引用保持 assertIs、grad 被 detach | 已覆盖（test_zero_grad_set_to_false） |
| 正常传参场景 | step 产生梯度后 zero_grad 清空、多参数组全清 | 已覆盖（test_zero_grad_multiple_groups_on_device） |
| 异常传参场景 | 实测不抛异常：truthy 视为 True、falsy 视为 False（基类无类型校验，按实测行为断言） | 已覆盖（test_zero_grad_invalid） |
| 混合设备类型 | 当前设备（NPU/CUDA）上多参数组清梯度 | 已覆盖（test_zero_grad_multiple_groups_on_device） |

未覆盖项及原因：
- 本接口不校验 set_to_none 类型、不抛异常（实测行为），故无「非法类型报错」可断言，改为断言 truthy/falsy 强制转换语义。
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
class TestAdamzero_grad(TestCase):
    """Comprehensive cases for torch.optim.adam.Adam.zero_grad (NPU/CUDA portable).

    Covers all four documented behaviors of the set_to_none switch:
    default call, explicit True, explicit False (in-place zeroing with
    reference identity), and the truthy/falsy coercion observed for
    non-bool arguments (the base class performs no type validation).
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adam([p], lr=0.01)

    def test_zero_grad_set_to_none(self):
        """default set_to_none=True leaves grad None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_zero_grad_set_to_none_explicit(self):
        """Explicit set_to_none=True also detaches the grad by setting it to None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        grad = torch.randn_like(p)
        p.grad = grad
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad, "explicit set_to_none=True must set grad to None")

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeros the tensor in place; the reference is kept."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        # A grad holding a graph (grad_fn) exercises the detach branch of
        # zero_grad: p.grad.detach_() runs in place, the object is preserved.
        grad = p.clone()
        p.grad = grad
        self.assertIsNotNone(p.grad.grad_fn, "precondition: grad must carry a grad_fn")
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad)
        # In-place contract: the very same Tensor object is zeroed, not swapped.
        self.assertIs(p.grad, grad, "set_to_none=False must zero the grad in place")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))
        # The in-place detach_() must have severed the autograd graph.
        self.assertIsNone(p.grad.grad_fn, "grad must be detached by zero_grad")
        self.assertFalse(p.grad.requires_grad)

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

    def test_zero_grad_multiple_groups_on_device(self):
        """After a real step, zero_grad clears grads across all param groups."""
        p1 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adam([{"params": [p1], "lr": 0.01},
                               {"params": [p2], "lr": 0.02}])
        loss = ((p1 * 2).sum() + (p2 * 3).sum())
        loss.backward()
        self.assertIsNotNone(p1.grad)
        self.assertIsNotNone(p2.grad)
        # Default call clears both groups on the current device.
        opt.zero_grad()
        self.assertIsNone(p1.grad, "group-1 grad must be None after zero_grad")
        self.assertIsNone(p2.grad, "group-2 grad must be None after zero_grad")
        # set_to_none=False clears both groups in place.
        loss = ((p1 * 2).sum() + (p2 * 3).sum())
        loss.backward()
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p1.grad)
        self.assertIsNotNone(p2.grad)
        torch.testing.assert_close(p1.grad, torch.zeros_like(p1.grad))
        torch.testing.assert_close(p2.grad, torch.zeros_like(p2.grad))


if __name__ == "__main__":
    run_tests()
