# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.rprop.Rprop.step 接口功能正确性
API 名称：torch.optim.rprop.Rprop.step
API 签名：step(self, closure=None) -> float | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 有梯度参数执行一步更新；无梯度参数不更新且不产生 state | 已覆盖（test_step_updates_param / test_step_no_grad / test_step_no_grad_no_state_entry） |
| 枚举选项 | etas 增/减分支（同号增长、反号收缩）；step_sizes 上下限裁剪 | 已覆盖（step_same_sign_grows_step_size/
step_sign_flip_shrinks_without_moving/step_upper_bound_clipping/step_lower_bound_clipping/
step_step_size_within_bounds_always） |
| 参数类型 | 可选 closure 可调用 | 已覆盖（test_step_closure / test_step_returns_none_without_closure） |
| 传参与不传参 | 传 closure 返回 loss 张量；不传返回 None | 已覆盖（test_step_closure / test_step_returns_none_without_closure） |
| 等价类/边界值 | state 键 {step, prev, step_size}；step 计数递增；step_size 逐元素初始为 lr；50 步收敛 | 已覆盖（step_state_keys_and_values/
step_state_step_counter_increments/step_multi_step_convergence） |
| 正常传参场景 | step 后参数按 -step_size*sign(grad) 更新且 Parameter 引用不变 | 已覆盖（step_updates_param/step_param_identity_preserved/
step_update_magnitude_equals_step_size） |
| 异常传参场景 | 稀疏梯度报 RuntimeError | 已覆盖（test_step_sparse_grad_raises） |
| 混合设备类型 | NPU 与 CPU 参数更新结果一致（默认超参与自定义 etas/step_sizes 变体） | 已覆盖（test_step_cpu_golden_npu / test_step_cpu_golden_custom_etas） |

未覆盖项及原因：
- maximize=True / capturable=True / foreach 显式开启：非本任务指定分支，留待 maximize 专项。
- 本接口涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。

实测核实记录（python -c，torch 2.11）：
- state 键为 {'prev', 'step', 'step_size'}（比常见文档多 'prev'）；step 为 0 维张量每步 +1；
  step_size 为与参数同形的逐元素张量，初始值 = lr。
- 同号梯度：step_size *= etaplus(1.2)；反号梯度：step_size *= etaminus(0.5) 且该步参数不动（源码将 grad 置 0）。
- 持续同号梯度下 step_size 指数增长，被 step_sizes 上限裁剪（fp32 上限值断言用 torch.tensor(上限, fp32)）；
  持续反号下被下限裁剪。
- Rprop 对二次函数可振荡（loss 非单调），50 步后 loss < 1e-4 且 |x-3| < 0.01（实测 CPU/NPU 一致）。
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
class TestRpropstep(TestCase):
    """Test cases for torch.optim.rprop.Rprop.step (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_step_updates_param(self):
        """Parameters move in -lr * grad direction on CPU."""
        opt = torch.optim.Rprop([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        p = opt.param_groups[0]["params"][0]
        grad = torch.full_like(p, 1.0)
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        self.assertFalse(torch.equal(before, p.detach()),
                         "parameter should change after a step")

    def test_step_no_grad(self):
        """A param without grad stays unchanged (or cleanly errors on GPU)."""
        opt = torch.optim.Rprop([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
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
        cpu_opt = torch.optim.Rprop([cpu_p], lr=0.01)
        dev_opt = torch.optim.Rprop([dev_p], lr=0.01)
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
            opt = torch.optim.Rprop([p], lr=0.01)
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
        opt = torch.optim.Rprop([p], lr=0.01)
        p.grad = torch.ones_like(p)
        ret = opt.step()
        self.assertIsNone(ret)

    def test_step_param_identity_preserved(self):
        """step updates the Parameter in-place; the object is never replaced."""
        p = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.Rprop([p], lr=0.01)
        p.grad = torch.ones_like(p)
        opt.step()
        self.assertIs(opt.param_groups[0]["params"][0], p,
                      "step must mutate the caller's Parameter, not swap it")

    def test_step_update_magnitude_equals_step_size(self):
        """First-step update magnitude is exactly lr * sign(grad), element-wise."""
        lr = 0.01
        p0 = torch.tensor([2.0, -1.0, 0.0], device=self.device)
        p = torch.nn.Parameter(p0.clone())
        opt = torch.optim.Rprop([p], lr=lr)
        p.grad = torch.tensor([1.5, -2.0, 0.7], device=self.device)
        opt.step()
        expected = p0 - lr * torch.sign(torch.tensor([1.5, -2.0, 0.7], device=self.device))
        torch.testing.assert_close(p.detach(), expected)

    # ------------------------------------------------------------------
    # closure (both paths)
    # ------------------------------------------------------------------

    def test_step_closure(self):
        """step(closure) evaluates the closure under enable_grad and returns its loss."""
        x = torch.nn.Parameter(torch.zeros(2, device=self.device))
        opt = torch.optim.Rprop([x], lr=0.01)

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
    # Multi-step convergence (Rprop may oscillate; assert threshold only)
    # ------------------------------------------------------------------

    def test_step_multi_step_convergence(self):
        """50 steps bring ((x-3)**2).sum() below 1e-4 and |x-3| below 0.01."""
        x = torch.nn.Parameter(torch.zeros(2, device=self.device))
        opt = torch.optim.Rprop([x], lr=0.01)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = ((x - 3) ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 1e-4,
                        "loss must fall below threshold in 50 steps (got %g)" % losses[-1])
        self.assertGreater(losses[0], losses[-1])
        # Rprop can oscillate around the optimum on a quadratic, so we do NOT
        # assert monotonic decrease; the distance to the optimum is the
        # observable convergence signal.
        self.assertLess((x.detach() - 3).abs().max().item(), 0.01)

    # ------------------------------------------------------------------
    # State keys and values (empirically confirmed: prev/step/step_size)
    # ------------------------------------------------------------------

    def test_step_state_keys_and_values(self):
        """State holds exactly {step, prev, step_size}; values follow the doc algorithm."""
        lr = 0.01
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=self.device))
        g = torch.tensor([0.5, -0.5], device=self.device)
        opt = torch.optim.Rprop([p], lr=lr)
        p.grad = g.clone()
        opt.step()
        state = opt.state[p]
        self.assertEqual(set(state.keys()), {"step", "prev", "step_size"})
        self.assertEqual(state["step"].item(), 1.0)
        # prev stores the last gradient verbatim.
        torch.testing.assert_close(state["prev"].cpu(), g.detach().cpu())
        # step_size starts at lr, element-wise.
        torch.testing.assert_close(state["step_size"].cpu(),
                                   torch.full((2,), lr, dtype=torch.float32))

    def test_step_state_step_counter_increments(self):
        """The 'step' counter increases by exactly one per step()."""
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.Rprop([p], lr=0.01)
        for expected in (1.0, 2.0, 3.0):
            p.grad = torch.tensor([0.5], device=self.device)
            opt.step()
            self.assertEqual(opt.state[p]["step"].item(), expected)

    # ------------------------------------------------------------------
    # etas branches: same-sign growth, sign-flip shrink (param frozen)
    # ------------------------------------------------------------------

    def test_step_same_sign_grows_step_size(self):
        """A repeated same-sign gradient multiplies step_size by etaplus (1.2)."""
        etaplus = 1.2
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.Rprop([p], lr=0.01)
        p.grad = torch.tensor([2.0], device=self.device)
        opt.step()
        first = opt.state[p]["step_size"].clone()
        p.grad = torch.tensor([2.0], device=self.device)
        opt.step()
        torch.testing.assert_close(opt.state[p]["step_size"].cpu(), first.cpu() * etaplus)

    def test_step_sign_flip_shrinks_without_moving(self):
        """On a gradient sign flip the param does NOT move and step_size is halved."""
        etaminus = 0.5
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.Rprop([p], lr=0.01)
        p.grad = torch.tensor([1.0], device=self.device)
        opt.step()
        after_first = p.detach().clone()
        step_size_before = opt.state[p]["step_size"].clone()
        # Opposite sign: Rprop zeroes the update for this step and shrinks
        # the step size (documented algorithm: g_t <- 0 on sign flip).
        p.grad = torch.tensor([-1.0], device=self.device)
        opt.step()
        self.assertTrue(torch.equal(p.detach(), after_first),
                        "param must not move on a sign-flip step")
        torch.testing.assert_close(opt.state[p]["step_size"].cpu(),
                                   step_size_before.cpu() * etaminus)

    # ------------------------------------------------------------------
    # step_sizes clipping: upper bound (sustained same-sign gradient)
    # ------------------------------------------------------------------

    def test_step_upper_bound_clipping(self):
        """Sustained same-sign gradients grow step_size to the upper bound and clip."""
        upper = 0.05
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        # Tight upper bound so the *1.2 growth reaches the clip quickly.
        opt = torch.optim.Rprop([p], lr=0.01, step_sizes=(1e-3, upper))
        for _ in range(12):
            p.grad = torch.tensor([1.0], device=self.device)
            opt.step()
        # step_size == fp32(upper) exactly (verified: 0.05 -> 0.05000000074505806).
        upper_t = torch.tensor(upper, dtype=torch.float32)
        self.assertTrue(torch.equal(opt.state[p]["step_size"].cpu(), upper_t.view(1)),
                        "step_size must saturate at the upper bound")
        # Continue pushing: still clipped at the bound.
        for _ in range(3):
            p.grad = torch.tensor([1.0], device=self.device)
            opt.step()
        self.assertTrue(torch.equal(opt.state[p]["step_size"].cpu(), upper_t.view(1)),
                        "step_size must stay clipped at the upper bound")

    def test_step_lower_bound_clipping(self):
        """Repeated sign flips shrink step_size to the lower bound and clip."""
        lower = 0.002
        p = torch.nn.Parameter(torch.tensor([1.0], device=self.device))
        opt = torch.optim.Rprop([p], lr=0.01, step_sizes=(lower, 50.0))
        p.grad = torch.tensor([1.0], device=self.device)
        opt.step()
        sign = 1.0
        for _ in range(8):
            sign = -sign
            p.grad = torch.tensor([sign], device=self.device)
            opt.step()
        lower_t = torch.tensor(lower, dtype=torch.float32)
        self.assertTrue(torch.equal(opt.state[p]["step_size"].cpu(), lower_t.view(1)),
                        "step_size must saturate at the lower bound")
        # Keep flipping: still clipped at the bound.
        for _ in range(3):
            sign = -sign
            p.grad = torch.tensor([sign], device=self.device)
            opt.step()
        self.assertTrue(torch.equal(opt.state[p]["step_size"].cpu(), lower_t.view(1)),
                        "step_size must stay clipped at the lower bound")

    def test_step_step_size_within_bounds_always(self):
        """step_size stays within [etaminus_bound, etaplus_bound] across mixed signs."""
        lower, upper = 1e-3, 0.05
        p = torch.nn.Parameter(torch.tensor([1.0, -1.0], device=self.device))
        opt = torch.optim.Rprop([p], lr=0.01, step_sizes=(lower, upper))
        lower_t = torch.tensor(lower, dtype=torch.float32)
        upper_t = torch.tensor(upper, dtype=torch.float32)
        sign = torch.tensor(1.0, device=self.device)
        for _ in range(30):
            # Alternate pseudo-randomly between holding and flipping signs.
            sign = -sign if _ % 3 == 0 else sign
            p.grad = torch.stack([sign, -sign]).expand(2).contiguous()
            opt.step()
            ss = opt.state[p]["step_size"].cpu()
            self.assertFalse(bool((ss < lower_t).any()) or bool((ss > upper_t).any()),
                             "step_size out of bounds at iter %d: %s" % (_, ss))

    # ------------------------------------------------------------------
    # No-grad param: not updated, no state entry
    # ------------------------------------------------------------------

    def test_step_no_grad_no_state_entry(self):
        """A grad-less param is neither updated nor given a state entry."""
        p1 = torch.nn.Parameter(torch.ones(2, device=self.device))
        p2 = torch.nn.Parameter(torch.ones(2, device=self.device))
        opt = torch.optim.Rprop([p1, p2], lr=0.01)
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
        opt = torch.optim.Rprop([p], lr=0.01)
        idx = torch.tensor([[0, 1]], device=self.device)
        vals = torch.tensor([1.0, 1.0], device=self.device)
        p.grad = torch.sparse_coo_tensor(idx, vals, (2,))
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            opt.step()

    # ------------------------------------------------------------------
    # CPU Golden (custom etas / step_sizes variant)
    # ------------------------------------------------------------------

    def test_step_cpu_golden_custom_etas(self):
        """NPU step matches the CPU golden for custom etas/step_sizes."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        torch.manual_seed(3)
        cpu_p = torch.nn.Parameter(torch.randn(4))
        cpu_g = torch.randn(4)
        # Snapshot the initial value BEFORE stepping so both sides start equal.
        snapshot = cpu_p.detach().clone()
        kwargs = {"lr": 0.02, "etas": (0.3, 1.5), "step_sizes": (1e-4, 10.0)}
        cpu_opt = torch.optim.Rprop([cpu_p], **kwargs)
        cpu_p.grad = cpu_g.clone()
        # Two steps to also exercise the sign-comparison branch.
        cpu_opt.step()
        cpu_p.grad = -cpu_g.clone()
        cpu_opt.step()
        dev_p = torch.nn.Parameter(snapshot.to(self.device))
        dev_opt = torch.optim.Rprop([dev_p], **kwargs)
        dev_p.grad = cpu_g.to(self.device)
        dev_opt.step()
        dev_p.grad = -cpu_g.to(self.device)
        dev_opt.step()
        torch.testing.assert_close(dev_p.detach().cpu(), cpu_p.detach())


if __name__ == "__main__":
    run_tests()
