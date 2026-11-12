# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.lbfgs.LBFGS.zero_grad 接口功能正确性
API 名称：torch.optim.lbfgs.LBFGS.zero_grad
API 签名：zero_grad(self, set_to_none: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数的 grad 被清空；真实 backward 产生的梯度同样被清空 | 已覆盖（test_zero_grad_default/test_zero_grad_after_real_backward） |
| 枚举选项 | set_to_none=True/False/省略（默认 True）三态分支 | 已覆盖（zero_grad_set_to_none_true/zero_grad_set_to_false/
zero_grad_default） |
| 参数类型 | set_to_none 声明为布尔；无类型校验（truthy/falsy） | 已覆盖（test_zero_grad_invalid） |
| 传参与不传参 | 显式 set_to_none=True 与省略参数默认调用 | 已覆盖（test_zero_grad_set_to_none_true/test_zero_grad_default） |
| 等价类/边界值 | set_to_none=False 为 in-place 置 0 且引用保持（assertIs 同一 grad 对象）；显式 True 置 None | 已覆盖（zero_grad_set_to_false/
zero_grad_set_to_none_true） |
| 正常传参场景 | 多参数组（LBFGS 仅允许构造后经 add_param_group 追加组，构造期多组抛 ValueError）两组同时清空；NPU/CPU 混合设备参数组 |
已覆盖（zero_grad_multiple_param_groups） |
| 异常传参场景 | zero_grad 不校验类型：truthy 视为 True、falsy 视为 False，不抛异常 | 已覆盖（test_zero_grad_invalid） |
| 混合设备类型 | 多参数组下 NPU 参数与 CPU 参数混合，两组 grad 均被正确清空（置 None / in-place 置 0） | 已覆盖（test_zero_grad_multiple_param_groups） |

未覆盖项及原因：
- LBFGS.step 相关梯度路径（step 内部经 closure 重算梯度）不属于本接口职责，仅在 test_zero_grad_after_real_backward
  中以真实 autograd backward 产生梯度后验证清空行为。
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
class TestLBFGSzero_grad(TestCase):
    """Test cases for torch.optim.lbfgs.LBFGS.zero_grad (NPU/CUDA portable).

    Covers all four set_to_none branches (explicit True / explicit False
    with in-place zeroing and reference preservation / omitted default /
    truthy-falsy values, which are not type-checked), plus real-autograd
    backward clearing and multi-param-group (mixed NPU/CPU) clearing.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=1.0):
        # LBFGS accepts a flat param list only: its __init__ rejects more
        # than one param group (ValueError), so keep a single group here.
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.LBFGS([p], lr=lr)

    def test_zero_grad_default(self):
        """default set_to_none=True leaves grad None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.randn_like(p)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_zero_grad_set_to_none_true(self):
        """Explicit set_to_none=True detaches grad by setting it to None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        grad_ref = torch.randn_like(p)
        p.grad = grad_ref
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad,
                          "explicit set_to_none=True must replace the grad with None")

    def test_zero_grad_set_to_false(self):
        """set_to_none=False zeros the tensor in place instead of None."""
        opt = self._make_optimizer()
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        grad_ref = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIsNotNone(p.grad)
        self.assertIs(p.grad, grad_ref,
                      "set_to_none=False must zero in place, keeping the same Tensor object")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad))

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

    def test_zero_grad_after_real_backward(self):
        """Gradients produced by a real autograd backward are cleared."""
        p = torch.nn.Parameter(torch.tensor([1.5, 2.5], device=self.device))
        opt = torch.optim.LBFGS([p], lr=0.5)
        loss = ((p - 0.5) ** 2).sum()
        loss.backward()
        self.assertIsNotNone(p.grad, "real backward must produce a grad to clear")
        opt.zero_grad()
        self.assertIsNone(p.grad,
                          "default zero_grad must clear the real backward grad")

        # in-place branch after another real backward
        loss = ((p - 0.5) ** 2).sum()
        loss.backward()
        grad_ref = p.grad
        opt.zero_grad(set_to_none=False)
        self.assertIs(p.grad, grad_ref,
                      "set_to_none=False must keep the same grad Tensor object")
        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad),
                                   msg="set_to_none=False must zero the real backward grad")

    def test_zero_grad_multiple_param_groups(self):
        """Both a device param group and a CPU param group are cleared together.

        LBFGS's __init__ rejects multiple param groups (ValueError), so the
        second group is attached via add_param_group; zero_grad itself walks
        all groups device-agnostically.
        """
        # boundary: constructing LBFGS with two groups is rejected upfront
        p_x = torch.nn.Parameter(torch.randn(2, device=self.device))
        p_y = torch.nn.Parameter(torch.randn(2, device=self.device))
        with self.assertRaisesRegex(ValueError, "parameter groups"):
            torch.optim.LBFGS([{"params": [p_x]}, {"params": [p_y]}], lr=1.0)

        opt = torch.optim.LBFGS([p_x], lr=1.0)
        p_cpu = torch.nn.Parameter(torch.randn(2))  # stays on CPU: mixed devices
        opt.add_param_group({"params": [p_cpu]})
        self.assertEqual(len(opt.param_groups), 2)

        p_x.grad = torch.ones_like(p_x)
        p_cpu.grad = torch.ones_like(p_cpu)
        gx_ref, gc_ref = p_x.grad, p_cpu.grad
        opt.zero_grad(set_to_none=False)
        self.assertIs(p_x.grad, gx_ref,
                      "device-group grad must be zeroed in place (same object)")
        self.assertIs(p_cpu.grad, gc_ref,
                      "CPU-group grad must be zeroed in place (same object)")
        torch.testing.assert_close(p_x.grad, torch.zeros_like(p_x.grad))
        torch.testing.assert_close(p_cpu.grad, torch.zeros_like(p_cpu.grad))

        # default branch clears both mixed-device groups to None
        p_x.grad = torch.ones_like(p_x)
        p_cpu.grad = torch.ones_like(p_cpu)
        opt.zero_grad()
        self.assertIsNone(p_x.grad)
        self.assertIsNone(p_cpu.grad)


if __name__ == "__main__":
    run_tests()
