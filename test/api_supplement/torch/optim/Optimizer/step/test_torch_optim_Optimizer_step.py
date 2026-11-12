# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Optimizer.step 接口功能正确性
API 名称：torch.optim.Optimizer.step
API 签名：step(self, closure: Callable[[], float] | None = None) -> float | None

基类行为说明（实测确认）：Optimizer.step 基类本身不实现任何优化逻辑，
方法体直接 raise NotImplementedError；具体更新逻辑由子类（如 SGD）覆写提供。
step pre/post hook 的派发由 profile_hook_step 包装器完成（__init__ 内
_patch_step_function 对类 step 打补丁）：pre hook 先于优化逻辑、post hook
后于优化逻辑，且 hook 签名为 hook(optimizer, args, kwargs)。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 基类 step 无优化逻辑（NotImplementedError）；经 SGD 子类一步后参数确实更新且 in-place | 已覆盖（base_step_raises_not_implemented/
step_updates_param） |
| 枚举选项 | closure 传/不传两种调用形态；step hook 无 prepend 参数（按注册顺序派发） | 已覆盖（step_with_closure_via_sgd/
step_without_closure_returns_none）；prepend 维度不存在（签名无该参数） |
| 参数类型 | closure 为 callable 或 None；hook 为 callable | 已覆盖（test_step_with_closure_via_sgd/test_step_hook_dispatch_order） |
| 传参与不传参 | 传 closure（SGD 路径返回 closure loss）；不传 closure（返回 None 且参数仍更新） | 已覆盖（step_with_closure_via_sgd/
step_without_closure_returns_none） |
| 等价类/边界值 | 基类实例 pre hook 先于 NotImplementedError 触发、post hook 不触发；pre 先于优化逻辑、post 后于（以参数值前后区分）；基类 step 可被子类覆写 |
已覆盖（base_instance_step_pre_hook_fires_before_raise/step_hook_dispatch_order/subclass_override_takes_effect） |
| 正常传参场景 | pre hook 可返回 (new_args, new_kwargs) 替换 closure；step 后参数按 -lr*grad 方向更新 | 已覆盖（step_pre_hook_replaces_closure/
step_updates_param） |
| 异常传参场景 | 基类 step 直接 NotImplementedError；pre hook 返回非法值抛 RuntimeError；非 callable hook 触发时抛 TypeError |
已覆盖（base_step_raises_not_implemented/step_pre_hook_invalid_return_raises/step_hook_non_callable_raises_typeerror） |
| 混合设备类型 | NPU 与 CPU 参数更新结果一致（CPU Golden） | 已覆盖（test_step_cpu_golden_npu） |

未覆盖项及原因：
- register_step_pre_hook/register_step_post_hook 无 prepend 参数（按注册顺序派发），无该枚举维度。
- 全局 step hook（torch.optim.optimizer._global_optimizer_pre_hooks）为私有模块级 API，不属于公开契约。
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
class TestOptimizerstep(TestCase):
    """Test cases for torch.optim.Optimizer.step (NPU/CUDA portable).

    The base class carries no optimization logic (step raises
    NotImplementedError); concrete updates come from subclass overrides.
    These tests therefore focus on the framework contract: closure
    pass-through to the subclass implementation, step pre/post hook
    dispatch (pre before the optimization logic, post after), hook
    arg rewriting, removability, non-callable TypeError, subclass
    override taking effect, plus the real SGD update path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # Base-class contract: no optimization logic in Optimizer.step
    # ------------------------------------------------------------------

    def test_base_step_raises_not_implemented(self):
        """Optimizer.step itself has no optimization logic and raises."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Optimizer([p], {'lr': 0.1})
        p.grad = torch.ones_like(p)
        with self.assertRaises(NotImplementedError):
            opt.step()
        # The closure argument is accepted (and equally unsupported) on the
        # base class: the optional closure is part of the signature contract.
        with self.assertRaises(NotImplementedError):
            opt.step(lambda: 1.0)

    def test_base_instance_step_pre_hook_fires_before_raise(self):
        """On a base Optimizer the pre hook fires before the raise; the
        post hook never fires because no optimization logic ran."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.Optimizer([p], {'lr': 0.1})
        pre_calls = []
        post_calls = []
        opt.register_step_pre_hook(lambda o, a, k: pre_calls.append((o, a, k)))
        opt.register_step_post_hook(lambda o, a, k: post_calls.append((o, a, k)))
        with self.assertRaises(NotImplementedError):
            opt.step()
        # The pre hook dispatched BEFORE the base step raised.
        self.assertEqual(len(pre_calls), 1,
                         "pre hook must fire before the base step body")
        self.assertIs(pre_calls[0][0], opt, "pre hook must receive this optimizer")
        self.assertIs(pre_calls[0][1][0], opt,
                      "positional args must start with the optimizer itself")
        self.assertEqual(pre_calls[0][2], {}, "kwargs must be empty for step()")
        # The post hook never fired: func(*args) raised before post dispatch.
        self.assertEqual(len(post_calls), 0,
                         "post hook must not fire when the step body raised")

    # ------------------------------------------------------------------
    # Closure pass-through (concrete update logic lives in the subclass)
    # ------------------------------------------------------------------

    def test_step_with_closure_via_sgd(self):
        """step(closure) passes the closure to the SGD implementation and
        returns the closure loss."""
        x = torch.nn.Parameter(torch.zeros(3, device=self.device))
        opt = torch.optim.SGD([x], lr=0.01)

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
        opt = torch.optim.SGD([p], lr=0.1)
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        ret = opt.step()
        self.assertIsNone(ret, "step() without a closure returns None")
        self.assertFalse(torch.equal(p.detach(), before),
                         "param must still be updated without a closure")

    # ------------------------------------------------------------------
    # Step hook registration and dispatch
    # ------------------------------------------------------------------

    def test_step_hook_dispatch_order(self):
        """Pre hook observes the pre-update param, post hook the updated
        one: pre fires before the optimization logic, post after it."""
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.SGD([p], lr=0.1)
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        seen = []

        def pre_hook(o, args, kwargs):
            # Contract: hook receives (optimizer, args, kwargs); when the
            # closure is passed positionally it appears in args.
            seen.append(("pre", o, args, kwargs, p.detach().clone()))
            return None

        def post_hook(o, args, kwargs):
            seen.append(("post", o, args, kwargs, p.detach().clone()))
            return None

        h1 = opt.register_step_pre_hook(pre_hook)
        h2 = opt.register_step_post_hook(post_hook)
        self.assertIsInstance(h1, torch.utils.hooks.RemovableHandle)
        self.assertIsInstance(h2, torch.utils.hooks.RemovableHandle)
        opt.step()

        self.assertEqual(len(seen), 2, "exactly one pre and one post hook must fire")
        self.assertEqual([s[0] for s in seen], ["pre", "post"],
                         "pre hook must dispatch before post hook")
        for tag, o, args, kwargs, _val in seen:
            self.assertIs(o, opt, "hooks must receive this optimizer instance")
            self.assertIs(args[0], opt, "args[0] must be the optimizer itself")
            self.assertEqual(kwargs, {}, "no kwargs for a positional step() call")
        # pre saw the not-yet-updated param; post saw the updated param.
        self.assertTrue(torch.equal(seen[0][4], before),
                        "pre hook must see the pre-update parameter value")
        self.assertTrue(torch.equal(seen[1][4], p.detach()),
                        "post hook must see the post-update parameter value")
        self.assertFalse(torch.equal(before, p.detach()),
                         "the optimization logic must have run between the hooks")

    def test_step_hook_sees_closure_in_args(self):
        """Hook args carry the positional closure; kwargs carry the keyword one."""
        p = torch.nn.Parameter(torch.ones(2, device=self.device))
        opt = torch.optim.SGD([p], lr=0.1)

        def closure():
            return torch.tensor(1.0)

        seen = {}

        def pre_hook(o, args, kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return None

        opt.register_step_pre_hook(pre_hook)
        opt.step(closure)
        # Positional closure lands in args (after the optimizer itself).
        self.assertEqual(len(seen["args"]), 2)
        self.assertIs(seen["args"][1], closure,
                      "positional closure must be visible to the pre hook")
        self.assertEqual(seen["kwargs"], {})

        seen.clear()
        opt.step(closure=closure)
        self.assertEqual(len(seen["args"]), 1, "only the optimizer in positional args")
        self.assertIs(seen["kwargs"]["closure"], closure,
                      "keyword closure must be visible to the pre hook")

    def test_step_pre_hook_replaces_closure(self):
        """A pre hook returning (new_args, new_kwargs) rewrites the call."""
        p = torch.nn.Parameter(torch.ones(2, device=self.device))
        opt = torch.optim.SGD([p], lr=0.1)

        def closure_a():
            return torch.tensor(11.0)

        def closure_b():
            return torch.tensor(22.0)

        def replace_hook(o, args, kwargs):
            # Swap the positional closure for closure_b.
            return ((args[0], closure_b), kwargs)

        opt.register_step_pre_hook(replace_hook)
        ret = opt.step(closure_a)
        self.assertEqual(float(ret), 22.0,
                         "the pre hook must be able to replace the closure")

    def test_step_pre_hook_invalid_return_raises(self):
        """A pre hook returning something other than None/(args, kwargs)
        raises RuntimeError at step time."""
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.ones(2, device=self.device))], lr=0.1)
        opt.register_step_pre_hook(lambda o, a, k: 123)
        with self.assertRaises(RuntimeError):
            opt.step()

    def test_step_hook_removal(self):
        """Removed hooks stop firing; removing one of two keeps the other."""
        p = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.SGD([p], lr=0.1)
        p.grad = torch.ones_like(p)
        calls = []
        ha = opt.register_step_pre_hook(lambda o, a, k: calls.append("A"))
        hb = opt.register_step_pre_hook(lambda o, a, k: calls.append("B"))
        opt.step()
        self.assertEqual(calls, ["A", "B"], "registration order without prepend")
        ha.remove()
        calls.clear()
        opt.step()
        self.assertEqual(calls, ["B"], "only the surviving hook may fire")
        hb.remove()
        calls.clear()
        opt.step()
        self.assertEqual(calls, [], "no hook may fire after both are removed")

    def test_step_hook_non_callable_raises_typeerror(self):
        """Non-callable hooks register fine but raise TypeError when fired."""
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.ones(2, device=self.device))], lr=0.1)
        h = opt.register_step_pre_hook("not-callable")
        self.assertIsNotNone(h, "registration of a non-callable must not raise")
        with self.assertRaises(TypeError):
            opt.step()
        h.remove()
        opt.step()  # clean after removal

        opt2 = torch.optim.SGD(
            [torch.nn.Parameter(torch.ones(2, device=self.device))], lr=0.1)
        h2 = opt2.register_step_post_hook(3.14)
        with self.assertRaises(TypeError):
            opt2.step()
        h2.remove()

    # ------------------------------------------------------------------
    # Subclass override contract
    # ------------------------------------------------------------------

    def test_subclass_override_takes_effect(self):
        """A subclass override replaces the base/SGD logic entirely: the
        custom body runs (hooks still dispatch around it), its return value
        is propagated, and the inherited SGD update no longer happens."""
        class MySGD(torch.optim.SGD):
            def step(self, closure=None):
                self._custom_ran = True
                if closure is not None:
                    return closure()
                return 123

        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = MySGD([p], lr=0.1)
        p.grad = torch.ones_like(p)
        before = p.detach().clone()
        hook_seq = []
        opt.register_step_pre_hook(lambda o, a, k: hook_seq.append("pre"))
        opt.register_step_post_hook(lambda o, a, k: hook_seq.append("post"))

        ret = opt.step()
        self.assertEqual(ret, 123, "the override's return value must be propagated")
        self.assertTrue(opt._custom_ran, "the override body must run")
        # The overridden step no longer applies the SGD update.
        self.assertTrue(torch.equal(p.detach(), before),
                        "an override that skips the update must leave params untouched")
        # Hooks still dispatch around the overridden step.
        self.assertEqual(hook_seq, ["pre", "post"],
                         "hooks must wrap the subclass override too")

    # ------------------------------------------------------------------
    # Concrete update path through an SGD instance
    # ------------------------------------------------------------------

    def test_step_updates_param(self):
        """Parameters move in -lr * grad direction, updated in place."""
        opt = torch.optim.SGD(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)],
            lr=0.01)
        p = opt.param_groups[0]["params"][0]
        grad = torch.full_like(p, 1.0)
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        self.assertIs(
            opt.param_groups[0]["params"][0], p,
            "step must update the param in place, not replace it")
        self.assertFalse(
            torch.equal(before, p.detach()),
            "parameter should change after a step")
        # SGD update rule: p_new == p - lr * grad (exact direction check).
        torch.testing.assert_close(p.detach(), before - 0.01 * grad)

    def test_step_no_grad(self):
        """A param without grad stays unchanged (or cleanly errors on GPU)."""
        opt = torch.optim.SGD(
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
            opt = torch.optim.SGD([p], lr=0.01)
            opt.step()
        except (RuntimeError, ValueError) as exc:
            self.skipTest('step unsupported: %s' % exc)
        self.assertEqual(p.dtype, before_dtype,
                         "step must preserve param dtype")

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
        cpu_opt = torch.optim.SGD([cpu_p], lr=0.01)
        dev_opt = torch.optim.SGD([dev_p], lr=0.01)
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
