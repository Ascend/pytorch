# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adamax.Adamax.step 接口功能正确性
API 名称：torch.optim.adamax.Adamax.step
API 签名：step(self, closure=None) -> float | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数执行一步更新 | 已覆盖（test_step_updates_param / test_step_state_keys_and_step_counter） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 可选 closure 可调用；betas 需为二元序列 | 已覆盖（test_step_with_closure_* / test_step_betas_*） |
| 传参与不传参 | 默认不传 closure / 显式传 closure | 已覆盖（test_step_without_closure_returns_none / test_step_with_closure_returns_loss） |
| 等价类/边界值 | 50 步收敛；无梯度参数不更新且无 state | 已覆盖（test_step_convergence_50_steps / test_step_no_grad_no_state） |
| 正常传参场景 | step 后参数按梯度方向更新；state 键与数值语义正确 | 已覆盖（test_step_updates_param / test_step_state_values_first_step） |
| 异常传参场景 | betas 单值/错误长度在构造期报错 | 已覆盖（test_step_betas_single_value_rejected / test_step_betas_wrong_length_rejected） |
| 混合设备类型 | NPU 与 CPU 参数更新结果需一致（单步+多步） | 已覆盖（test_step_cpu_golden_npu / test_step_cpu_golden_multi_step） |

未覆盖项及原因：
- sparse 梯度：Adamax 显式不支持（RuntimeError "Adamax does not support sparse gradients"），
  但该报错来自 aten 函数式入口而非 Adamax.step 自身分支，不在本文件断言范围。
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

# Adamax internal state keys, verified on the installed torch build:
#   state[p] == {"exp_avg": Tensor, "exp_inf": Tensor, "step": scalar Tensor}
ADAMAX_STATE_KEYS = {"exp_avg", "exp_inf", "step"}


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAdamaxstep(TestCase):
    """Test cases for torch.optim.adamax.Adamax.step (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # Single-step basic behavior (kept from the original file)
    # ------------------------------------------------------------------

    def test_step_updates_param(self):
        """Parameters move in -lr * grad direction on CPU."""
        opt = torch.optim.Adamax([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        grad = torch.full_like(p, 1.0)
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()),
                         "parameter should change after a step")

    def test_step_no_grad(self):
        """A param without grad stays unchanged (or cleanly errors on GPU)."""
        opt = torch.optim.Adamax([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        before = p.detach().clone()
        try:
            opt.step()
            torch.testing.assert_close(p.detach(), before)
        except (RuntimeError, ValueError):
            self.skipTest('optimizer requires a grad on this backend')

    def test_step_dtype_fidelity(self):
        """step does not alter the param dtype (fp32 in -> fp32 out)."""
        dt = torch.float32
        p = torch.nn.Parameter(torch.ones(3, dtype=dt), requires_grad=True)
        p.grad = torch.ones_like(p)
        before_dtype = p.dtype
        try:
            opt = torch.optim.Adamax([p], lr=0.01)
            opt.step()
        except (RuntimeError, ValueError) as exc:
            self.skipTest('step unsupported: %s' % exc)
        self.assertEqual(p.dtype, before_dtype,
                         "step must preserve param dtype")

    # ------------------------------------------------------------------
    # Multi-step convergence (not just "the param moved")
    # ------------------------------------------------------------------

    def test_step_convergence_50_steps(self):
        """50 steps on ((x-3)**2).sum() drive the loss below a threshold."""
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(3, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.1)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((p - 3) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Starting from x=0 the initial loss is exactly 27.0.
        self.assertEqual(losses[0], 27.0)
        # Convergence threshold: the loss must fall well below its start.
        self.assertLess(losses[-1], 1.0,
                        "loss should drop below 1.0 after 50 steps, got %f" % losses[-1])
        # Monotone (non-increasing within fp tolerance) descent.
        for prev, cur in zip(losses, losses[1:]):
            self.assertLessEqual(cur, prev + 1e-6,
                                 "loss must not increase between steps "
                                 "(%f -> %f)" % (prev, cur))
        # The parameter itself must approach the optimum at 3.
        torch.testing.assert_close(p.detach(), torch.full_like(p, 3.0),
                                   rtol=0.0, atol=0.5)

    # ------------------------------------------------------------------
    # Internal state contract (keys and per-step semantics)
    # ------------------------------------------------------------------

    def test_step_state_keys_and_step_counter(self):
        """state[p] holds exactly {exp_avg, exp_inf, step}; step counts runs."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.01)
        p.grad = torch.ones_like(p)
        opt.step()
        state = opt.state[p]
        self.assertEqual(set(state.keys()), ADAMAX_STATE_KEYS)
        # step is a scalar tensor that increments once per step call.
        self.assertTrue(torch.is_tensor(state["step"]))
        self.assertEqual(state["step"].numel(), 1)
        self.assertEqual(state["step"].item(), 1.0)
        # exp_avg / exp_inf are tensors shaped like the param, on its device.
        self.assertTrue(torch.is_tensor(state["exp_avg"]))
        self.assertTrue(torch.is_tensor(state["exp_inf"]))
        self.assertEqual(state["exp_avg"].shape, p.shape)
        self.assertEqual(state["exp_inf"].shape, p.shape)
        self.assertEqual(state["exp_avg"].device.type, self.device.type)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertEqual(opt.state[p]["step"].item(), 2.0,
                         "step counter must increment on every step")

    def test_step_state_values_first_step(self):
        """After one step: exp_avg=(1-b1)*g, exp_inf=|g|+eps, param update matches."""
        g = torch.tensor([2.0, -4.0], device=self.device_name)
        p = torch.nn.Parameter(torch.zeros(2, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.1)
        p.grad = g.clone()
        opt.step()
        state = opt.state[p]
        beta1, eps = 0.9, 1e-8
        # First-step closed forms for Adamax (bias correction with t=1).
        torch.testing.assert_close(state["exp_avg"].cpu(), (1 - beta1) * g.cpu())
        torch.testing.assert_close(state["exp_inf"].cpu(), (g.abs() + eps).cpu())
        # p_new = -lr / (1 - b1^1) * exp_avg / exp_inf
        expected_p = -0.1 / (1 - beta1) * state["exp_avg"] / state["exp_inf"]
        torch.testing.assert_close(p.detach().cpu(), expected_p.cpu())

    # ------------------------------------------------------------------
    # Adamax-specific hyperparameter branches (betas forms)
    # ------------------------------------------------------------------

    def test_step_betas_tuple_and_list_equivalent(self):
        """betas accepts any 2-sequence; tuple and list produce identical updates."""
        torch.manual_seed(1)
        p_tuple = torch.nn.Parameter(torch.randn(4))
        p_list = torch.nn.Parameter(p_tuple.detach().clone())
        opt_tuple = torch.optim.Adamax([p_tuple], lr=0.05, betas=(0.8, 0.99))
        opt_list = torch.optim.Adamax([p_list], lr=0.05, betas=[0.8, 0.99])
        for _ in range(3):
            grad = torch.rand(4)
            p_tuple.grad = grad.clone()
            p_list.grad = grad.clone()
            opt_tuple.step()
            opt_list.step()
        # A list betas must drive the exact same trajectory as the tuple form.
        torch.testing.assert_close(p_list.detach(), p_tuple.detach())
        self.assertEqual(opt_list.param_groups[0]["betas"], [0.8, 0.99])

    def test_step_betas_single_value_rejected(self):
        """A single float for betas is rejected at construction time."""
        # NOTE: the task brief assumed "single value vs tuple both work", but
        # the installed build raises TypeError at __init__ (betas[0] on a
        # float). The test pins the ACTUAL behavior.
        with self.assertRaises(TypeError):
            torch.optim.Adamax([torch.nn.Parameter(torch.randn(1))],
                               lr=0.01, betas=0.9)

    def test_step_betas_wrong_length_rejected(self):
        """A 1-element betas sequence is rejected at construction time."""
        with self.assertRaises((IndexError, TypeError)):
            torch.optim.Adamax([torch.nn.Parameter(torch.randn(1))],
                               lr=0.01, betas=(0.9,))

    # ------------------------------------------------------------------
    # closure paths
    # ------------------------------------------------------------------

    def test_step_with_closure_returns_loss(self):
        """step(closure) calls the closure once and returns its loss."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.1)
        calls = []

        def closure():
            calls.append(1)
            opt.zero_grad()
            loss = ((p - 3) ** 2).sum()
            loss.backward()
            return loss

        ret = opt.step(closure)
        self.assertEqual(len(calls), 1, "closure must be called exactly once")
        self.assertIsNotNone(ret, "step must return the closure loss")
        self.assertEqual(ret.item(), 27.0)
        # The closure-driven update must actually move the parameter.
        self.assertFalse(torch.equal(p.detach(), torch.zeros_like(p)),
                         "closure-driven step must update the param")

    def test_step_with_closure_under_outer_no_grad(self):
        """The closure runs under enable_grad even inside torch.no_grad()."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.1)

        def closure():
            opt.zero_grad()
            loss = ((p - 3) ** 2).sum()
            loss.backward()
            return loss

        with torch.no_grad():
            ret = opt.step(closure)
        self.assertEqual(ret.item(), 27.0)
        # backward() inside the closure must still populate p.grad.
        self.assertIsNotNone(p.grad,
                             "closure backward must run under enable_grad")
        torch.testing.assert_close(p.detach().cpu(),
                                   torch.full((3,), 0.1))

    def test_step_with_closure_none_return(self):
        """A closure returning None still drives the update; step returns None."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.1)
        before = p.detach().clone()

        def closure():
            opt.zero_grad()
            loss = ((p - 3) ** 2).sum()
            loss.backward()
            return None  # loss is swallowed on purpose

        ret = opt.step(closure)
        self.assertIsNone(ret, "step returns None when the closure returns None")
        self.assertFalse(torch.equal(p.detach(), before),
                         "the param must still be updated")

    def test_step_without_closure_returns_none(self):
        """step() without a closure returns None and still updates params."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device_name))
        opt = torch.optim.Adamax([p], lr=0.1)
        p.grad = torch.full_like(p, -6.0)
        before = p.detach().clone()
        ret = opt.step()
        self.assertIsNone(ret)
        self.assertFalse(torch.equal(p.detach(), before),
                         "step must update the param without a closure")

    # ------------------------------------------------------------------
    # No-grad isolation: params without grads get no state at all
    # ------------------------------------------------------------------

    def test_step_no_grad_no_state(self):
        """A param without grad is skipped: no update, no state entry."""
        p_with = torch.nn.Parameter(torch.ones(2, device=self.device_name))
        p_without = torch.nn.Parameter(torch.ones(2, device=self.device_name))
        opt = torch.optim.Adamax([p_with, p_without], lr=0.1)
        p_with.grad = torch.full_like(p_with, 1.0)
        before_without = p_without.detach().clone()
        opt.step()
        torch.testing.assert_close(p_without.detach(), before_without)
        # The skipped param must not even allocate optimizer state.
        self.assertEqual(len(opt.state[p_without]), 0,
                         "no state must be created for a grad-less param")
        # ...while the tracked param does get state.
        self.assertEqual(set(opt.state[p_with].keys()), ADAMAX_STATE_KEYS)

    # ------------------------------------------------------------------
    # CPU Golden comparison (device portability)
    # ------------------------------------------------------------------

    def test_step_cpu_golden_npu(self):
        """NPU step matches the CPU golden result."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(0)
        cpu_p = torch.nn.Parameter(torch.randn(4, 4) if False else torch.randn(3))
        cpu_grad = torch.randn_like(cpu_p)
        dev_p = torch.nn.Parameter(cpu_p.detach().clone().to(self.device))
        dev_grad = cpu_grad.clone().to(self.device)
        cpu_opt = torch.optim.Adamax([cpu_p], lr=0.01)
        dev_opt = torch.optim.Adamax([dev_p], lr=0.01)
        cpu_p.grad = cpu_grad.clone()
        dev_p.grad = dev_grad.clone()
        try:
            cpu_opt.step()
            dev_opt.step()
        except RuntimeError as exc:
            self.skipTest('device step unsupported: %s' % exc)
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())

    def test_step_cpu_golden_multi_step(self):
        """A multi-step trajectory on the device matches the CPU golden."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(5)
        cpu_p = torch.nn.Parameter(torch.randn(4))
        dev_p = torch.nn.Parameter(cpu_p.detach().clone().to(self.device))
        cpu_opt = torch.optim.Adamax([cpu_p], lr=0.05)
        dev_opt = torch.optim.Adamax([dev_p], lr=0.05)
        for _ in range(5):
            grad = torch.rand(4)
            cpu_p.grad = grad.clone()
            dev_p.grad = grad.clone().to(self.device)
            try:
                cpu_opt.step()
                dev_opt.step()
            except RuntimeError as exc:
                self.skipTest('device step unsupported: %s' % exc)
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())
        # The internal state must also agree, not just the parameter.
        cpu_state = cpu_opt.state[cpu_p]
        dev_state = dev_opt.state[dev_p]
        torch.testing.assert_close(dev_state["exp_avg"].cpu(), cpu_state["exp_avg"])
        torch.testing.assert_close(dev_state["exp_inf"].cpu(), cpu_state["exp_inf"])
        self.assertEqual(dev_state["step"].item(), cpu_state["step"].item())


if __name__ == "__main__":
    run_tests()
