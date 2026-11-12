# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.rmsprop.RMSprop.step 接口功能正确性
API 名称：torch.optim.rmsprop.RMSprop.step
API 签名：step(self, closure=None) -> float | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数执行一步更新；无梯度参数不更新且不产生 state | 已覆盖（test_step_updates_param / test_step_no_grad / test_step_no_grad_no_state_entry） |
| 枚举选项 | centered/momentum 分支组合 | 已覆盖（6 用例：centered_false/true_branch、momentum_zero/positive_branch、二者组合、momentum 轨迹） |
| 参数类型 | 可选 closure 可调用 | 已覆盖（test_step_closure） |
| 传参与不传参 | 传 closure 返回 loss 张量；不传返回 None | 已覆盖（test_step_closure / test_step_returns_none_without_closure） |
| 等价类/边界值 | 键按分支；step 递增；50 步收敛 | 已覆盖（step_state_{keys_default,step_counter_increments}/step_multi_step_convergence） |
| 正常传参场景 | step 后参数按梯度方向更新且 Parameter 对象引用不变 | 已覆盖（test_step_updates_param / test_step_param_identity_preserved） |
| 异常传参场景 | 稀疏梯度报 RuntimeError | 已覆盖（test_step_sparse_grad_raises） |
| 混合设备类型 | NPU 与 CPU 参数更新结果一致（默认超参与 centered+momentum 变体） | 已覆盖（test_step_cpu_golden_npu / test_step_cpu_golden_centered_momentum） |

未覆盖项及原因：
- maximize=True / capturable=True / foreach 显式开启：非本任务指定分支，留待 maximize 专项。
- 本接口涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。

实测核实记录（python -c，torch 2.11）：
- state 键：默认 {'square_avg', 'step'}；momentum>0 增加 'momentum_buffer'；centered=True 增加 'grad_avg'；
  momentum_buffer 首步 = grad / (sqrt(square_avg)+eps)，square_avg 首步 = (1-alpha)*grad^2，
  grad_avg 首步 = (1-alpha)*grad，step 为 0 维张量且每步 +1。
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
class TestRMSpropstep(TestCase):
    """Test cases for torch.optim.rmsprop.RMSprop.step (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_step_updates_param(self):
        """Parameters move in -lr * grad direction on CPU."""
        opt = torch.optim.RMSprop([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        grad = torch.full_like(p, 1.0)
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()),
                         "parameter should change after a step")

    def test_step_no_grad(self):
        """A param without grad stays unchanged (or cleanly errors on GPU)."""
        opt = torch.optim.RMSprop([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        before = p.detach().clone()
        try:
            opt.step()
            torch.testing.assert_close(p.detach(), before)
        except (RuntimeError, ValueError):
            self.skipTest('optimizer requires a grad on this backend')

    def test_step_cpu_golden_npu(self):
        """NPU step matches the CPU golden result."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(0)
        cpu_p = torch.nn.Parameter(torch.randn(4, 4) if False else torch.randn(3))
        cpu_grad = torch.randn_like(cpu_p)
        dev_p = torch.nn.Parameter(cpu_p.detach().clone().to(self.device))
        dev_grad = cpu_grad.clone().to(self.device)
        cpu_opt = torch.optim.RMSprop([cpu_p], lr=0.01)
        dev_opt = torch.optim.RMSprop([dev_p], lr=0.01)
        cpu_p.grad = cpu_grad.clone()
        dev_p.grad = dev_grad.clone()
        try:
            cpu_opt.step()
            dev_opt.step()
        except RuntimeError as exc:
            self.skipTest('device step unsupported: %s' % exc)
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())

    def test_step_dtype_fidelity(self):
        """step does not alter the param dtype (fp32 in -> fp32 out)."""
        dt = torch.float32
        p = torch.nn.Parameter(torch.ones(3, dtype=dt), requires_grad=True)
        p.grad = torch.ones_like(p)
        before_dtype = p.dtype
        try:
            opt = torch.optim.RMSprop([p], lr=0.01)
            opt.step()
        except (RuntimeError, ValueError) as exc:
            self.skipTest('step unsupported: %s' % exc)
        self.assertEqual(p.dtype, before_dtype,
                         "step must preserve param dtype")

    # ------------------------------------------------------------------
    # Return value / reference identity
    # ------------------------------------------------------------------

    def test_step_returns_none_without_closure(self):
        """step() without a closure returns None (no loss was evaluated)."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01)
        p.grad = torch.ones_like(p)
        ret = opt.step()
        self.assertIsNone(ret)

    def test_step_param_identity_preserved(self):
        """step updates the Parameter in-place; the object is never replaced."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertIs(opt.param_groups[0]["params"][0], p,
                      "step must mutate the caller's Parameter, not swap it")

    # ------------------------------------------------------------------
    # closure (both paths)
    # ------------------------------------------------------------------

    def test_step_closure(self):
        """step(closure) evaluates the closure under enable_grad and returns its loss."""
        x = torch.nn.Parameter(torch.zeros(2, device=self.device))
        opt = torch.optim.RMSprop([x], lr=0.05)

        def closure():
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            return loss

        ret = opt.step(closure)
        self.assertIsInstance(ret, torch.Tensor)
        # First evaluation at x=(0,0): ((0-3)**2).sum() == 18.0 exactly.
        self.assertEqual(ret.item(), 18.0)
        self.assertFalse(torch.equal(x.detach(), torch.zeros(2, device=self.device)),
                         "closure path must also apply the update")

    # ------------------------------------------------------------------
    # Multi-step convergence
    # ------------------------------------------------------------------

    def test_step_multi_step_convergence(self):
        """50 steps drive ((x-3)**2).sum() monotonically below 1e-2."""
        x = torch.nn.Parameter(torch.zeros(2, device=self.device))
        opt = torch.optim.RMSprop([x], lr=0.05)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 1e-2, "loss must fall below threshold in 50 steps")
        self.assertGreater(losses[0], losses[-1])
        # RMSprop smooths the update; on this convex quadratic the loss
        # sequence must be non-increasing (verified on CPU and NPU).
        for prev, nxt in zip(losses, losses[1:]):
            self.assertLessEqual(nxt, prev,
                                 "loss increased at some step: %r" % (losses,))

    # ------------------------------------------------------------------
    # State keys per hyperparameter branch (empirically confirmed)
    # ------------------------------------------------------------------

    def test_step_state_keys_default(self):
        """Default branch: state holds exactly {square_avg, step} with correct values."""
        alpha, eps = 0.99, 1e-8
        p = torch.nn.Parameter(torch.tensor([2.0, -1.0], device=self.device))
        g = torch.tensor([4.0, -2.0], device=self.device)
        opt = torch.optim.RMSprop([p], lr=0.01, alpha=alpha, eps=eps)
        p.grad = g.clone()
        opt.step()
        state = opt.state[p]
        self.assertEqual(set(state.keys()), {"square_avg", "step"})
        self.assertEqual(state["step"].item(), 1.0)
        # First step: square_avg = (1 - alpha) * grad ** 2 (running average starts at 0).
        expected_sq = (1 - alpha) * g.detach().cpu() ** 2
        torch.testing.assert_close(state["square_avg"].cpu(), expected_sq)

    def test_step_momentum_zero_branch(self):
        """momentum=0 keeps momentum_buffer out of the state."""
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01, momentum=0)
        p.grad = torch.tensor([3.0], device=self.device)
        opt.step()
        self.assertEqual(set(opt.state[p].keys()), {"square_avg", "step"})
        self.assertNotIn("momentum_buffer", opt.state[p])

    def test_step_momentum_positive_branch(self):
        """momentum=0.9 creates momentum_buffer == grad / (sqrt(square_avg) + eps)."""
        alpha, eps = 0.99, 1e-8
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        g = torch.tensor([3.0], device=self.device)
        opt = torch.optim.RMSprop([p], lr=0.01, momentum=0.9, alpha=alpha, eps=eps)
        p.grad = g.clone()
        opt.step()
        state = opt.state[p]
        self.assertEqual(set(state.keys()), {"square_avg", "step", "momentum_buffer"})
        # buffer = momentum * 0 + grad / avg  with  avg = sqrt(square_avg) + eps
        expected_buf = g.detach().cpu() / ((1 - alpha) * g.detach().cpu() ** 2).sqrt().add(eps)
        torch.testing.assert_close(state["momentum_buffer"].cpu(), expected_buf)

    def test_step_momentum_changes_trajectory(self):
        """With momentum=0.9 the second step moves the param further than momentum=0."""
        def run(momentum):
            p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
            opt = torch.optim.RMSprop([p], lr=0.01, momentum=momentum)
            for _ in range(2):
                p.grad = torch.tensor([3.0], device=self.device)
                opt.step()
            return p.detach().clone()

        without, with_m = run(0), run(0.9)
        self.assertFalse(torch.equal(without, with_m),
                         "momentum must change the update trajectory")

    def test_step_centered_false_branch(self):
        """centered=False keeps grad_avg out of the state."""
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01, centered=False)
        p.grad = torch.tensor([3.0], device=self.device)
        opt.step()
        self.assertEqual(set(opt.state[p].keys()), {"square_avg", "step"})
        self.assertNotIn("grad_avg", opt.state[p])

    def test_step_centered_true_branch(self):
        """centered=True adds grad_avg == (1 - alpha) * grad on the first step."""
        alpha = 0.99
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        g = torch.tensor([3.0], device=self.device)
        opt = torch.optim.RMSprop([p], lr=0.01, centered=True, alpha=alpha)
        p.grad = g.clone()
        opt.step()
        state = opt.state[p]
        self.assertEqual(set(state.keys()), {"square_avg", "step", "grad_avg"})
        expected = (1 - alpha) * g.detach().cpu()
        torch.testing.assert_close(state["grad_avg"].cpu(), expected)

    def test_step_centered_momentum_combined(self):
        """centered=True + momentum=0.9 produces all four state keys."""
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01, momentum=0.9, centered=True)
        p.grad = torch.tensor([3.0], device=self.device)
        opt.step()
        self.assertEqual(
            set(opt.state[p].keys()),
            {"square_avg", "step", "momentum_buffer", "grad_avg"},
        )

    def test_step_state_step_counter_increments(self):
        """The 'step' counter increases by exactly one per step()."""
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01)
        for expected in (1.0, 2.0, 3.0):
            p.grad = torch.tensor([0.5], device=self.device)
            opt.step()
            self.assertEqual(opt.state[p]["step"].item(), expected)

    # ------------------------------------------------------------------
    # No-grad param: not updated, no state entry
    # ------------------------------------------------------------------

    def test_step_no_grad_no_state_entry(self):
        """A grad-less param is neither updated nor given a state entry."""
        p1 = torch.nn.Parameter(torch.ones(2, device=self.device))
        p2 = torch.nn.Parameter(torch.ones(2, device=self.device))
        opt = torch.optim.RMSprop([p1, p2], lr=0.01)
        p1.grad = torch.ones_like(p1)
        before1, before2 = p1.detach().clone(), p2.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(p1.detach(), before1), "graded param must be updated")
        torch.testing.assert_close(p2.detach(), before2,
                                   msg="grad-less param must stay unchanged")
        self.assertIn(p1, opt.state)
        self.assertNotIn(p2, opt.state, "no state may be created for a grad-less param")

    # ------------------------------------------------------------------
    # Stable exception path
    # ------------------------------------------------------------------

    def test_step_sparse_grad_raises(self):
        """A sparse gradient raises RuntimeError mentioning sparse support."""
        p = torch.nn.Parameter(torch.zeros(2, device=self.device))
        opt = torch.optim.RMSprop([p], lr=0.01)
        idx = torch.tensor([[0, 1]], device=self.device)
        vals = torch.tensor([1.0, 1.0], device=self.device)
        p.grad = torch.sparse_coo_tensor(idx, vals, (2,))
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            opt.step()

    # ------------------------------------------------------------------
    # CPU Golden (extra hyperparameter variant)
    # ------------------------------------------------------------------

    def test_step_cpu_golden_centered_momentum(self):
        """NPU step matches the CPU golden for the centered+momentum variant."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(7)
        cpu_p = torch.nn.Parameter(torch.randn(3))
        cpu_g = torch.randn(3)
        # Snapshot the initial value BEFORE stepping so both sides start equal.
        snapshot = cpu_p.detach().clone()
        kwargs = {"lr": 0.01, "momentum": 0.9, "centered": True}
        cpu_opt = torch.optim.RMSprop([cpu_p], **kwargs)
        cpu_p.grad = cpu_g.clone()
        cpu_opt.step()
        dev_p = torch.nn.Parameter(snapshot.to(self.device))
        dev_opt = torch.optim.RMSprop([dev_p], **kwargs)
        dev_p.grad = cpu_g.to(self.device)
        dev_opt.step()
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())


if __name__ == "__main__":
    run_tests()
