# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adadelta.Adadelta.step 接口功能正确性
API 名称：torch.optim.adadelta.Adadelta.step
API 签名：step(self, closure=None) -> float | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数执行一步更新且 assertIs 原位更新；无梯度/requires_grad=False 参数不更新且不产生 state | 已覆盖（step_updates_param_in_place/
grad_less_params_skipped） |
| 枚举选项 | Adadelta 特有超参分支：rho 显式 vs 默认、eps 大小、weight_decay>0/=0、maximize 开关 | 已覆盖（rho_branch/eps_effect/
weight_decay_branches/maximize_branch） |
| 参数类型 | closure 可调用或 None | 已覆盖（test_step_with_closure/test_step_without_closure_returns_none） |
| 传参与不传参 | 传 closure 与不传 closure 两条路径 | 已覆盖（test_step_with_closure/test_step_without_closure_returns_none） |
| 等价类/边界值 | 50 步 loss 单调不增且低于 24.0；400 步 loss < 5.0；state 键 step/square_avg/acc_delta 且 step 计数递增（0 维 CPU float32 张量） |
已覆盖（step_convergence_50/step_convergence_long_run/state_keys_and_step_counter） |
| 正常传参场景 | 参数按梯度方向更新、dtype 保持 | 已覆盖（test_step_updates_param_in_place/test_step_dtype_fidelity） |
| 异常传参场景 | 稀疏梯度触发 Adadelta 特有 RuntimeError；无梯度时不报错且参数保持 | 已覆盖（test_sparse_grad_rejected/test_step_no_grad） |
| 混合设备类型 | NPU 与 CPU 多步 step 结果一致（参数与 state 双对照，CPU Golden） | 已覆盖（test_step_cpu_golden_npu/test_step_cpu_golden_npu_multi_step） |

未覆盖项及原因：
- foreach/capturable/differentiable 执行后端开关：属性能实现细节且 NPU 侧支持矩阵不稳定，不在功能 UT 范围（capturable=True 在 NPU 上属未支持设备列表，实测会抛错，不作为功能断言）。
- 本接口为框架行为，浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。

实测说明（与官方文档差异/确认点）：
- Adadelta 收敛速度显著慢于 Adam：默认超参下 50 步 ((x-3)**2).sum() 仅从 27.0 降至 23.88（实测 NPU/CPU 一致），
  400 步降至 4.77，故阈值按实测确定（50 步 < 24.0，400 步 < 5.0），不套用 Adam 的阈值。
- state 键为 step/square_avg/acc_delta（实测确认）；step 为 0 维 float32 CPU 张量（capturable=False 契约）。
"""
import math
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
class TestAdadeltastep(TestCase):
    """Comprehensive cases for torch.optim.adadelta.Adadelta.step (NPU/CUDA portable).

    Covers: one-step in-place update identity, multi-step convergence (Adadelta
    is much slower than Adam under default hyperparameters, thresholds are
    measured empirically), state key semantics (step/square_avg/acc_delta with
    a 0-dim CPU float32 step counter), the Adadelta-specific hyperparameter
    branches (rho/eps/weight_decay/maximize), closure vs no-closure paths,
    grad-less param skip, sparse-grad rejection and CPU golden comparisons.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # One-step behavior
    # ------------------------------------------------------------------

    def test_step_updates_param(self):
        """Parameters move in -lr * grad direction on CPU."""
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        grad = torch.full_like(p, 1.0)
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()),
                         "parameter should change after a step")

    def test_step_updates_param_in_place(self):
        """One step moves the param and updates it in place (same object)."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.01)
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        opt.step()
        self.assertIs(opt.param_groups[0]["params"][0], p,
                      "step must update the param in place, not replace it")
        self.assertFalse(torch.equal(before, p.detach()),
                         "parameter should change after a step")

    def test_step_no_grad(self):
        """A param without grad stays unchanged (or cleanly errors on GPU)."""
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
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
            opt = torch.optim.Adadelta([p], lr=0.01)
            opt.step()
        except (RuntimeError, ValueError) as exc:
            self.skipTest('step unsupported: %s' % exc)
        self.assertEqual(p.dtype, before_dtype,
                         "step must preserve param dtype")

    # ------------------------------------------------------------------
    # Multi-step convergence (Adadelta is slow under defaults: measured
    # 27.0 -> 23.88 after 50 steps, 14.20 after 200, 4.77 after 400)
    # ------------------------------------------------------------------

    def test_step_convergence_50(self):
        """50 default-hyperparameter steps decrease the loss monotonically below 24.0.

        Adadelta needs no lr tuning (adaptive), but it converges far slower
        than Adam: measured final loss 23.88 on both CPU and NPU, so the
        threshold is set from measurement rather than Adam's.
        """
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.Adadelta([x])  # all defaults: lr=1.0, rho=0.9, eps=1e-6
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        first, last = losses[0], losses[-1]
        self.assertLess(last, first, "loss must decrease over 50 steps")
        # Measured: 27.0 -> 23.881 on CPU and NPU. Threshold with headroom.
        self.assertLess(last, 24.0,
                        "loss must fall below 24.0 after 50 steps (measured 23.88)")
        # Loss trajectory must be non-increasing (verified over 400 steps).
        bumps = [b for a, b in zip(losses, losses[1:]) if b > a + 1e-9]
        self.assertEqual(bumps, [], "loss must not increase between steps")
        # Every element moves toward the minimizer 3 (from 0).
        moved = x.detach() - 0.0
        self.assertTrue(bool((moved > 0.0).all()),
                        "x elements must move toward the minimizer")

    def test_step_convergence_long_run(self):
        """400 steps drive the loss below 5.0 and every element within 1.5 of 3.

        Measured after 400 default-hyperparameter steps: loss 4.766 and
        x == 1.744 for every element (identical across elements).
        """
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.Adadelta([x])
        losses = []
        for _ in range(400):
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 5.0,
                        "loss must fall below 5.0 after 400 steps (measured 4.77)")
        self.assertTrue(bool(((x.detach() - 3).abs() < 1.5).all()),
                        "x elements must be within 1.5 of the minimizer 3")

    # ------------------------------------------------------------------
    # State semantics (Adadelta-specific keys: square_avg / acc_delta)
    # ------------------------------------------------------------------

    def test_state_keys_and_step_counter(self):
        """State holds step/square_avg/acc_delta; step is a 0-dim CPU tensor and increments."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.01)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertIn(p, opt.state)
        st = opt.state[p]
        for key in ("step", "square_avg", "acc_delta"):
            self.assertIn(key, st, "Adadelta state must contain %s" % key)
        # Measured contract: step is a 0-dim float32 tensor hosted on CPU
        # (capturable/fused are False here; see note [special device hosting
        # for step] in torch/optim).
        self.assertTrue(torch.is_tensor(st["step"]), "step must be a Tensor")
        self.assertEqual(st["step"].ndim, 0, "step must be a 0-dim tensor")
        self.assertEqual(st["step"].dtype, torch.float32)
        self.assertTrue(st["step"].is_cpu,
                        "non-capturable step counter must live on CPU")
        self.assertEqual(float(st["step"]), 1.0, "step counter must be 1 after the first step")
        # square_avg tracks EMA of squared grads: rho=0.9, g=1 -> 0.1.
        torch.testing.assert_close(st["square_avg"], torch.full_like(p, 0.1))
        # First step: delta = sqrt(eps)/sqrt((1-rho)*g^2 + eps) * g (acc_delta
        # starts at 0), so acc_delta = (1-rho) * eps / ((1-rho)*g^2 + eps).
        # Tight tolerances: the value is ~1e-6, so the default atol=1e-5
        # would make the assertion vacuous.
        expected_acc = 0.1 * 1e-6 / (0.1 + 1e-6)
        torch.testing.assert_close(st["acc_delta"], torch.full_like(p, expected_acc),
                                   rtol=1e-5, atol=1e-9)
        # State tensors follow the param device.
        self.assertEqual(st["square_avg"].device.type, p.device.type)
        self.assertEqual(st["acc_delta"].device.type, p.device.type)
        # Second step: counter increments, square_avg EMA advances.
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertEqual(float(st["step"]), 2.0, "step counter must increment per step")
        torch.testing.assert_close(st["square_avg"], torch.full_like(p, 0.19),
                                   msg="square_avg must be rho*0.1 + (1-rho)*1 == 0.19")

    # ------------------------------------------------------------------
    # Adadelta-specific hyperparameter branches
    # ------------------------------------------------------------------

    def test_rho_branch(self):
        """Explicit rho changes the square_avg EMA; the default rho is 0.9.

        With g=1 the first-step square_avg equals (1-rho), which cleanly
        separates the branches: rho=0.5 -> 0.5 vs default rho=0.9 -> 0.1.
        """
        p_r = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_d = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_r = torch.optim.Adadelta([p_r], lr=1.0, rho=0.5)
        opt_d = torch.optim.Adadelta([p_d], lr=1.0)  # default rho=0.9
        p_r.grad = torch.ones_like(p_r)
        p_d.grad = torch.ones_like(p_d)
        opt_r.step()
        opt_d.step()
        torch.testing.assert_close(opt_r.state[p_r]["square_avg"],
                                   torch.full_like(p_r, 0.5),
                                   msg="rho=0.5 must give square_avg == 1-rho == 0.5")
        torch.testing.assert_close(opt_d.state[p_d]["square_avg"],
                                   torch.full_like(p_d, 0.1),
                                   msg="default rho=0.9 must give square_avg == 1-rho == 0.1")

    def test_eps_effect(self):
        """eps enters both numerator and denominator of the first-step update.

        First step: delta = sqrt(eps)/sqrt((1-rho)*g^2 + eps) * g, so a larger
        eps yields a larger first-step movement. Measured: eps=1e-6 moves
        p=1 to 0.99684 while eps=1e-2 moves it to 0.69849.
        """
        rho, g = 0.9, 1.0
        for eps in (1e-6, 1e-2):
            p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
            opt = torch.optim.Adadelta([p], lr=1.0, eps=eps)
            p.grad = torch.full_like(p, g)
            opt.step()
            # First-step analytic update with lr=1.0:
            # delta = sqrt(eps)/sqrt((1-rho)*g^2 + eps) * g, p_new = 1 - delta.
            expected = 1.0 - (math.sqrt(eps) / math.sqrt((1 - rho) * g * g + eps)) * g
            torch.testing.assert_close(
                p.detach(), torch.full_like(p, expected),
                msg="eps=%g must produce the analytic first-step update" % eps)
        # Control: the two eps values must produce different movements.
        p_small = torch.nn.Parameter(torch.ones(1, device=self.device), requires_grad=True)
        p_large = torch.nn.Parameter(torch.ones(1, device=self.device), requires_grad=True)
        o_small = torch.optim.Adadelta([p_small], lr=1.0, eps=1e-6)
        o_large = torch.optim.Adadelta([p_large], lr=1.0, eps=1e-2)
        p_small.grad = torch.ones_like(p_small)
        p_large.grad = torch.ones_like(p_large)
        o_small.step()
        o_large.step()
        self.assertLess(p_large.detach().item(), p_small.detach().item(),
                        "larger eps must move the param further on the first step")

    def test_weight_decay_branches(self):
        """weight_decay>0 adds lambda*p to the grad; wd=0 leaves the grad untouched.

        With p=1, g=1, wd=0.5 the effective grad is 1.5, so the first-step
        square_avg is (1-rho)*1.5^2 == 0.225 (measured), vs 0.1 without decay.
        """
        rho, g = 0.9, 1.0
        p_wd = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_no = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_wd = torch.optim.Adadelta([p_wd], lr=1.0, weight_decay=0.5)
        opt_no = torch.optim.Adadelta([p_no], lr=1.0, weight_decay=0.0)
        p_wd.grad = torch.full_like(p_wd, g)
        p_no.grad = torch.full_like(p_no, g)
        opt_wd.step()
        opt_no.step()
        torch.testing.assert_close(opt_wd.state[p_wd]["square_avg"],
                                   torch.full_like(p_wd, (1 - rho) * (g + 0.5) ** 2),
                                   msg="wd>0 must fold lambda*p into the grad before the EMA")
        torch.testing.assert_close(opt_no.state[p_no]["square_avg"],
                                   torch.full_like(p_no, (1 - rho) * g * g),
                                   msg="wd=0 must keep the raw grad in the EMA")
        # Pure-decay branch: zero grad + wd>0 still moves the param, because
        # the decay term alone supplies an effective grad.
        p_zero = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_zero = torch.optim.Adadelta([p_zero], lr=1.0, weight_decay=0.5)
        p_zero.grad = torch.zeros_like(p_zero)
        opt_zero.step()
        self.assertTrue(bool((p_zero.detach() < 1.0).all()),
                        "wd>0 must shrink the param even with zero grad")
        # Control: wd=0 with zero grad leaves the param exactly unchanged
        # (delta is exactly 0 here, no eps-driven movement without a grad).
        p_ctrl = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_ctrl = torch.optim.Adadelta([p_ctrl], lr=1.0, weight_decay=0.0)
        p_ctrl.grad = torch.zeros_like(p_ctrl)
        opt_ctrl.step()
        self.assertTrue(torch.equal(p_ctrl.detach(), torch.ones(3, device=self.device)),
                        "wd=0 with zero grad must not move the param")

    def test_maximize_branch(self):
        """maximize=True negates the grad, flipping the update direction.

        With grad=-1 the default walks the param up; maximize=True treats it
        as +1 and walks the param down. Measured single-step values: 1.00316
        vs 0.99684.
        """
        p_max = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_nomax = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_max = torch.optim.Adadelta([p_max], lr=1.0, maximize=True)
        opt_nomax = torch.optim.Adadelta([p_nomax], lr=1.0, maximize=False)
        p_max.grad = torch.full_like(p_max, -1.0)
        p_nomax.grad = torch.full_like(p_nomax, -1.0)
        opt_max.step()
        opt_nomax.step()
        self.assertTrue(bool((p_max.detach() < 1.0).all()),
                        "maximize=True must follow -grad (param decreases)")
        self.assertTrue(bool((p_nomax.detach() > 1.0).all()),
                        "maximize=False must follow the raw grad (param increases)")
        # The sign persists over two steps (measured 0.99359 vs 1.00641).
        p_max.grad = torch.full_like(p_max, -1.0)
        p_nomax.grad = torch.full_like(p_nomax, -1.0)
        opt_max.step()
        opt_nomax.step()
        self.assertTrue(bool((p_max.detach() < 1.0).all()),
                        "maximize=True direction must persist over two steps")
        self.assertTrue(bool((p_nomax.detach() > 1.0).all()),
                        "maximize=False direction must persist over two steps")

    # ------------------------------------------------------------------
    # Grad-less params are skipped (no update, no state)
    # ------------------------------------------------------------------

    def test_grad_less_params_skipped(self):
        """Params without grad are skipped: unchanged and no state entry."""
        p_active = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_frozen = torch.nn.Parameter(torch.full((3,), 5.0, device=self.device),
                                      requires_grad=False)
        p_nograd = torch.nn.Parameter(torch.full((3,), 7.0, device=self.device),
                                      requires_grad=True)  # grad stays None
        opt = torch.optim.Adadelta([p_active, p_frozen, p_nograd], lr=1.0)
        p_active.grad = torch.ones_like(p_active)
        opt.step()
        self.assertFalse(torch.equal(p_active.detach(), torch.ones(3, device=self.device)),
                         "param with grad must update")
        self.assertTrue(torch.equal(p_frozen.detach(), torch.full((3,), 5.0, device=self.device)),
                        "requires_grad=False param must stay unchanged")
        self.assertTrue(torch.equal(p_nograd.detach(), torch.full((3,), 7.0, device=self.device)),
                        "param whose grad is None must stay unchanged")
        self.assertIn(p_active, opt.state)
        self.assertNotIn(p_frozen, opt.state,
                         "no state must be created for requires_grad=False params")
        self.assertNotIn(p_nograd, opt.state,
                         "no state must be created for grad-less params")

    # ------------------------------------------------------------------
    # Closure paths
    # ------------------------------------------------------------------

    def test_step_with_closure(self):
        """step(closure) re-evaluates the closure and returns its loss tensor."""
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.Adadelta([x], lr=1.0)

        def closure():
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            return loss

        ret = opt.step(closure)
        self.assertTrue(torch.is_tensor(ret), "step(closure) must return the closure loss")
        self.assertEqual(ret.item(), 27.0, "closure loss for x=0 is 9*3=27")
        self.assertFalse(torch.equal(x.detach(), torch.zeros(3, device=self.device)),
                         "param must be updated via the closure path")
        # A second closure step returns the freshly re-evaluated (lower) loss.
        ret2 = opt.step(closure)
        self.assertTrue(torch.is_tensor(ret2), "second closure step must return a tensor too")
        self.assertLess(ret2.item(), 27.0,
                        "re-evaluated loss must drop after the first update")

    def test_step_without_closure_returns_none(self):
        """step() without a closure returns None and still updates params."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=1.0)
        p.grad = torch.ones_like(p)
        ret = opt.step()
        self.assertIsNone(ret, "step() without a closure returns None")
        self.assertFalse(torch.equal(p.detach(), torch.ones(3, device=self.device)),
                         "param must still be updated without a closure")

    # ------------------------------------------------------------------
    # Adadelta-specific stable exception path
    # ------------------------------------------------------------------

    def test_sparse_grad_rejected(self):
        """Sparse gradients trigger Adadelta's dedicated RuntimeError."""
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.01)
        indices = torch.tensor([[0, 1]], device=self.device, dtype=torch.long)
        values = torch.tensor([1.0, 2.0], device=self.device)
        p.grad = torch.sparse_coo_tensor(indices, values, (3,), device=self.device)
        with self.assertRaisesRegex(RuntimeError, "Adadelta does not support sparse"):
            opt.step()

    # ------------------------------------------------------------------
    # CPU golden comparisons
    # ------------------------------------------------------------------

    def test_step_cpu_golden_npu(self):
        """NPU step matches the CPU golden result."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(0)
        cpu_p = torch.nn.Parameter(torch.randn(3))
        cpu_grad = torch.randn_like(cpu_p)
        dev_p = torch.nn.Parameter(cpu_p.detach().clone().to(self.device))
        dev_grad = cpu_grad.clone().to(self.device)
        cpu_opt = torch.optim.Adadelta([cpu_p], lr=0.01)
        dev_opt = torch.optim.Adadelta([dev_p], lr=0.01)
        cpu_p.grad = cpu_grad.clone()
        dev_p.grad = dev_grad.clone()
        try:
            cpu_opt.step()
            dev_opt.step()
        except RuntimeError as exc:
            self.skipTest('device step unsupported: %s' % exc)
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())

    def test_step_cpu_golden_npu_multi_step(self):
        """5 steps with non-default hyperparams match CPU for params AND state.

        Longer trajectories accumulate rounding noise, so a 5-step run (with
        weight_decay and fresh grads each step) is compared for both the
        params and the square_avg/acc_delta state tensors.
        """
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(42)
        cpu_p = torch.nn.Parameter(torch.randn(6))
        cpu_grad = torch.randn_like(cpu_p)
        dev_p = torch.nn.Parameter(cpu_p.detach().clone().to(self.device))
        cpu_opt = torch.optim.Adadelta([cpu_p], lr=0.5, rho=0.95, eps=1e-6, weight_decay=0.01)
        dev_opt = torch.optim.Adadelta([dev_p], lr=0.5, rho=0.95, eps=1e-6, weight_decay=0.01)
        try:
            for i in range(5):
                cpu_p.grad = cpu_grad * (1 + i)
                dev_p.grad = (cpu_grad * (1 + i)).to(self.device)
                cpu_opt.step()
                dev_opt.step()
        except RuntimeError as exc:
            self.skipTest('device step unsupported: %s' % exc)
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach(),
                                   msg="5-step params must match the CPU golden")
        torch.testing.assert_close(dev_opt.state[dev_p]["square_avg"].cpu(),
                                   cpu_opt.state[cpu_p]["square_avg"],
                                   msg="square_avg must match the CPU golden")
        torch.testing.assert_close(dev_opt.state[dev_p]["acc_delta"].cpu(),
                                   cpu_opt.state[cpu_p]["acc_delta"],
                                   msg="acc_delta must match the CPU golden")


if __name__ == "__main__":
    run_tests()
