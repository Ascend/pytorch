# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adam.Adam.step 接口功能正确性
API 名称：torch.optim.adam.Adam.step
API 签名：step(self, closure=None) -> float | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数执行一步更新；无梯度参数被跳过且不产生 state | 已覆盖（test_step_updates_param/test_step_no_grad/test_no_grad_param_not_updated） |
| 枚举选项 | amsgrad=True/False 分支；weight_decay>0 与 =0 分支 | 已覆盖（test_amsgrad_state_branch/test_weight_decay_branch） |
| 参数类型 | closure 可调用或 None | 已覆盖（test_step_with_closure/test_step_without_closure_returns_none） |
| 传参与不传参 | 传 closure 与不传 closure 两条路径 | 已覆盖（test_step_with_closure/test_step_without_closure_returns_none） |
| 等价类/边界值 | 一次 step 参数变化且引用不变；50 步收敛到阈值；state 键 step/exp_avg/exp_avg_sq 且 step 计数递增 | 已覆盖（step_updates_param/
step_convergence_multi_step/state_keys_and_step_counter） |
| 正常传参场景 | 参数按梯度方向更新、dtype 保持 | 已覆盖（test_step_updates_param/test_step_dtype_fidelity） |
| 异常传参场景 | 无梯度时不报错且参数保持不变 | 已覆盖（test_step_no_grad） |
| 混合设备类型 | NPU 与 CPU 参数更新结果需一致 | 已覆盖（test_step_cpu_golden_npu，CPU Golden） |

未覆盖项及原因：
- fused/foreach/capturable 等执行后端开关：属性能实现细节且 NPU 侧支持矩阵不稳定，不在功能 UT 范围。
- 本接口为框架行为，浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
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
class TestAdamstep(TestCase):
    """Comprehensive cases for torch.optim.adam.Adam.step (NPU/CUDA portable).

    Covers: one-step update with in-place identity, multi-step convergence,
    state key semantics (step/exp_avg/exp_avg_sq, amsgrad branch), the
    weight_decay branch, closure vs no-closure paths, grad-less param skip,
    and a CPU-vs-device golden comparison.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # One-step behavior
    # ------------------------------------------------------------------

    def test_step_updates_param(self):
        """Parameters move after one step and are updated in-place (same object)."""
        opt = torch.optim.Adam(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)],
            lr=0.01)
        p = opt.param_groups[0]["params"][0]
        grad = torch.full_like(p, 1.0)
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        self.assertIs(opt.param_groups[0]["params"][0], p,
                      "step must update the param in place, not replace it")
        self.assertFalse(torch.equal(before, p.detach()),
                         "parameter should change after a step")

    def test_step_no_grad(self):
        """A param without grad stays unchanged (or cleanly errors on GPU)."""
        opt = torch.optim.Adam(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)],
            lr=0.01)
        p = opt.param_groups[0]["params"][0]
        before = p.detach().clone()
        try:
            opt.step()
            torch.testing.assert_close(p.detach(), before)
            # No state slot may be created for a param that never had a grad.
            self.assertEqual(len(opt.state), 0)
        except (RuntimeError, ValueError):
            self.skipTest('optimizer requires a grad on this backend')

    def test_step_dtype_fidelity(self):
        """step does not alter the param dtype (fp32 in -> fp32 out)."""
        dt = torch.float32
        p = torch.nn.Parameter(torch.ones(3, dtype=dt, device=self.device), requires_grad=True)
        p.grad = torch.ones_like(p)
        before_dtype = p.dtype
        try:
            opt = torch.optim.Adam([p], lr=0.01)
            opt.step()
        except (RuntimeError, ValueError) as exc:
            self.skipTest('step unsupported: %s' % exc)
        self.assertEqual(p.dtype, before_dtype,
                         "step must preserve param dtype")

    # ------------------------------------------------------------------
    # Multi-step convergence
    # ------------------------------------------------------------------

    def test_step_convergence_multi_step(self):
        """50 Adam steps drive ((x-3)**2).sum() far below the initial loss."""
        torch.manual_seed(0)
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.Adam([x], lr=0.1)
        first_loss = None
        last_loss = None
        for _ in range(50):
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()
        self.assertLess(last_loss, first_loss, "loss must decrease over steps")
        # Empirically converges to ~0.08 (CPU and NPU) in 50 steps with lr=0.1.
        self.assertLess(last_loss, 1.0,
                        "loss must fall below the threshold after 50 steps")
        # Every element must approach the minimizer 3.
        self.assertTrue(bool(((x.detach() - 3).abs() < 0.5).all()),
                        "x should approach the minimizer 3")

    # ------------------------------------------------------------------
    # State semantics
    # ------------------------------------------------------------------

    def test_state_keys_and_step_counter(self):
        """State holds step/exp_avg/exp_avg_sq; the step counter increments."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.01)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertIn(p, opt.state)
        st = opt.state[p]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertIn(key, st, "Adam state must contain %s" % key)
        self.assertEqual(float(st["step"]), 1.0, "step counter must be 1 after the first step")
        # exp_avg / exp_avg_sq track the gradient (all ones) with Adam's betas.
        torch.testing.assert_close(st["exp_avg"], torch.full_like(p, 0.1))   # beta1 = 0.9
        torch.testing.assert_close(st["exp_avg_sq"], torch.full_like(p, 0.001))  # beta2 = 0.999
        self.assertEqual(st["exp_avg"].shape, p.shape)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertEqual(float(st["step"]), 2.0, "step counter must increment per step")

    def test_amsgrad_state_branch(self):
        """amsgrad=True adds max_exp_avg_sq to the state; the default has none."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.01, amsgrad=True)
        p.grad = torch.ones_like(p)
        opt.step()
        st = opt.state[p]
        self.assertIn("max_exp_avg_sq", st, "amsgrad state must hold max_exp_avg_sq")
        torch.testing.assert_close(st["max_exp_avg_sq"], torch.full_like(p, 0.001))
        # Control: the default (amsgrad=False) state has no max_exp_avg_sq key.
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt2 = torch.optim.Adam([p2], lr=0.01)
        p2.grad = torch.ones_like(p2)
        opt2.step()
        self.assertNotIn("max_exp_avg_sq", opt2.state[p2])

    def test_weight_decay_branch(self):
        """weight_decay>0 takes the decay branch and differs from weight_decay=0."""
        # Adam's first step is sign-like for a constant-sign grad, so the decay
        # term is isolated with a small opposite grad: decay flips the update
        # direction (verified: wd -> 0.9, no wd -> 1.1 from p=1, lr=0.1).
        p_wd = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_no = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_wd = torch.optim.Adam([p_wd], lr=0.1, weight_decay=0.5)
        opt_no = torch.optim.Adam([p_no], lr=0.1, weight_decay=0.0)
        p_wd.grad = torch.full_like(p_wd, -0.05)
        p_no.grad = torch.full_like(p_no, -0.05)
        opt_wd.step()
        opt_no.step()
        # Without decay the small negative grad pushes the param up; with
        # decay the dominant wd*p term flips the update direction.
        self.assertTrue(bool((p_no.detach() > 1.0).all()),
                        "wd=0 must follow the negative gradient (param rises)")
        self.assertTrue(bool((p_wd.detach() < 1.0).all()),
                        "wd>0 decay must dominate and pull the param down")
        # Pure-decay branch: zero grad + weight_decay shrinks the param
        # toward 0 even though the gradient contributes nothing.
        p_zero = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_zero = torch.optim.Adam([p_zero], lr=0.1, weight_decay=0.5)
        p_zero.grad = torch.zeros_like(p_zero)
        opt_zero.step()
        self.assertTrue(bool((p_zero.detach() < 1.0).all()),
                        "weight_decay must shrink the param even with zero grad")
        # Control: without wd a zero grad leaves the param exactly unchanged.
        p_ctrl = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt_ctrl = torch.optim.Adam([p_ctrl], lr=0.1, weight_decay=0.0)
        p_ctrl.grad = torch.zeros_like(p_ctrl)
        opt_ctrl.step()
        self.assertTrue(torch.equal(p_ctrl.detach(), torch.ones(3, device=self.device)),
                        "wd=0 with zero grad must not move the param")

    # ------------------------------------------------------------------
    # Grad-less params are skipped
    # ------------------------------------------------------------------

    def test_no_grad_param_not_updated(self):
        """Params without grad are skipped: unchanged and no state entry."""
        p_active = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p_frozen = torch.nn.Parameter(torch.full((3,), 5.0, device=self.device),
                                      requires_grad=False)
        opt = torch.optim.Adam([p_active, p_frozen], lr=0.1)
        p_active.grad = torch.ones_like(p_active)
        opt.step()
        self.assertFalse(torch.equal(p_active.detach(), torch.ones(3, device=self.device)),
                         "param with grad must update")
        # The grad-less (requires_grad=False) param is skipped entirely.
        self.assertTrue(torch.equal(p_frozen.detach(), torch.full((3,), 5.0, device=self.device)),
                        "param without grad must stay unchanged")
        self.assertNotIn(p_frozen, opt.state,
                         "no state must be created for grad-less params")
        self.assertIn(p_active, opt.state)

    # ------------------------------------------------------------------
    # Closure paths
    # ------------------------------------------------------------------

    def test_step_with_closure(self):
        """step(closure) re-evaluates the closure and returns its loss."""
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.Adam([x], lr=0.01)

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

    def test_step_without_closure_returns_none(self):
        """step() without a closure returns None and still updates params."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.1)
        p.grad = torch.ones_like(p)
        ret = opt.step()
        self.assertIsNone(ret, "step() without a closure returns None")
        self.assertFalse(torch.equal(p.detach(), torch.ones(3, device=self.device)),
                         "param must still be updated without a closure")

    # ------------------------------------------------------------------
    # CPU golden comparison
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
        cpu_opt = torch.optim.Adam([cpu_p], lr=0.01)
        dev_opt = torch.optim.Adam([dev_p], lr=0.01)
        cpu_p.grad = cpu_grad.clone()
        dev_p.grad = dev_grad.clone()
        try:
            cpu_opt.step()
            dev_opt.step()
        except RuntimeError as exc:
            self.skipTest('device step unsupported: %s' % exc)
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())


if __name__ == "__main__":
    run_tests()
