# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adadelta.Adadelta.register_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.adadelta.Adadelta.register_state_dict_post_hook
API 签名：register_state_dict_post_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则默认状态 | 已覆盖（test_hook_trigger/test_hook_fires_after_packing） |
| 枚举选项 | prepend=False/True 控制触发次序（[B,A] prepend 双 hook + [C,D] append 对照） | 已覆盖（test_hook_prepend_ordering） |
| 参数类型 | hook(opt, state_dict) 可返回新 dict 覆盖结果、返回 None 保留原结果、in-place 修改可见 | 已覆盖（hook_return_dict_replaces_output/
hook_return_none_keeps_original/hook_inplace_mutation_visible） |
| 传参与不传参 | 显式 hook/prepend 与默认调用；handle.remove() 停止触发、二删一留 | 已覆盖（hook_removed_stops_firing/
hook_remove_one_of_two/hook_trigger） |
| 等价类/边界值 | 触发位置在 state_dict 打包之后：params 为 int 索引、组内携带超参 lr/rho/eps、step 后 state 键 square_avg/acc_delta 且 step 计数递增 |
已覆盖（hook_fires_after_packing/hook_multi_group_packing） |
| 正常传参场景 | 注册后在 state_dict() 返回值生成后确实触发（assertIs 优化器同一性） | 已覆盖（test_hook_fires_after_packing） |
| 异常传参场景 | 非 callable 注册不抛、state_dict() 触发 TypeError | 已覆盖（test_hook_non_callable_raises_on_state_dict/test_hook_trigger） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 混合设备类型：本接口为纯框架回调注册 API，无张量迁移语义；打包后 state 内张量（square_avg/acc_delta）位于参数所在设备、step 计数张量位于 CPU（实测），仅做数值断言。
- pre hook（register_state_dict_pre_hook）为另一个 API 的契约，不在本文件范围。

实测说明（与前批 hook 契约结论一致，直接采用）：
- post hook 返回值生效：返回 dict 时替换 state_dict() 输出；返回 None 时保留原输出；in-place 修改对输出可见。
- 非 callable 注册本身不抛异常，调用 state_dict() 时触发 TypeError: 'str' object is not callable。
- prepend=True 时后注册者先触发：双 hook 顺序 [B, A]；prepend=False（默认）为注册顺序 [C, D]。
- 打包后 params 为 int 索引（[0]、[[0],[1]]），state 以 int 为键；Adadelta state 键为 step/square_avg/acc_delta。
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
class TestAdadeltaregister_state_dict_post_hook(TestCase):
    """Comprehensive cases for torch.optim.adadelta.Adadelta.register_state_dict_post_hook (NPU/CUDA portable).

    Covers: the fires-after-packing contract (int param indices, Adadelta
    hyperparams in packed groups, square_avg/acc_delta state keys with the
    step counter), return-value semantics (dict replaces / None keeps /
    in-place visible), prepend ordering with a two-hook [B,A] plus [C,D]
    control, removal semantics (stop firing, remove-one-keep-other), the
    non-callable TypeError path and multi-group packing.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # Existing basic trigger test (kept)
    # ------------------------------------------------------------------

    def test_hook_trigger(self):
        """Hook is called at the documented lifecycle point."""
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        seen = []
        opt.register_state_dict_post_hook(lambda o, sd: seen.append((o, sd)))
        sd = opt.state_dict()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], opt)
        self.assertIsInstance(seen[0][1], dict)

        # post-hook may return a replacement dict
        def inject(o, sd):
            sd["extra"] = 42
            return sd

        opt.register_state_dict_post_hook(inject)
        out = opt.state_dict()
        self.assertEqual(out.get("extra"), 42)

        # removable + removal stops firing
        calls = []
        h = opt.register_state_dict_post_hook(lambda o, sd: calls.append(1))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls), 1)

        opt.register_state_dict_post_hook("not-callable")
        with self.assertRaises(TypeError):
            opt.state_dict()

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = torch.optim.Adadelta([torch.nn.Parameter(torch.randn(3), requires_grad=True)], lr=0.01)
        handle = opt.register_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()

    # ------------------------------------------------------------------
    # Fires-after-packing contract
    # ------------------------------------------------------------------

    def test_hook_fires_after_packing(self):
        """The hook receives the packed dict: int indices, Adadelta hyperparams, Adadelta state keys.

        Packing contract (measured): params are replaced by int order indices,
        the group still carries Adadelta's lr/rho/eps hyperparams, and after a
        step the state dict holds square_avg/acc_delta plus the step counter.
        """
        p = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([p], lr=0.7, rho=0.8, eps=1e-5)
        p.grad = torch.ones_like(p)
        opt.step()
        seen = []
        opt.register_state_dict_post_hook(lambda o, sd: seen.append((o, sd)))
        out = opt.state_dict()
        self.assertEqual(len(seen), 1, "the hook must fire exactly once per state_dict call")
        hook_opt, packed = seen[0]
        self.assertIs(hook_opt, opt, "the hook must receive this very optimizer")
        self.assertIs(packed, out, "with None returned, the packed dict itself is the output")
        # Params are packed as int order indices, not Tensor objects.
        group = packed["param_groups"][0]
        self.assertEqual(group["params"], [0])
        for idx in group["params"]:
            self.assertIsInstance(idx, int, "packed params must be int indices")
        # The packed group carries Adadelta hyperparams verbatim.
        self.assertEqual(group["lr"], 0.7)
        self.assertEqual(group["rho"], 0.8)
        self.assertEqual(group["eps"], 1e-5)
        self.assertEqual(group["weight_decay"], 0)
        # The packed state holds Adadelta-specific keys with the step counter.
        self.assertEqual(sorted(packed["state"].keys()), [0])
        st = packed["state"][0]
        self.assertEqual(sorted(st.keys()), ["acc_delta", "square_avg", "step"])
        self.assertEqual(float(st["step"]), 1.0,
                         "packed step counter must be 1 after one step")
        # square_avg after one step with g=1, rho=0.8 is (1-rho) == 0.2
        # (the packed tensor stays on the param device; compare on CPU).
        torch.testing.assert_close(st["square_avg"].cpu(),
                                   torch.full((2,), 0.2),
                                   msg="packed square_avg must be (1-rho)*g^2 == 0.2")

    def test_hook_multi_group_packing(self):
        """With two groups the hook sees [[0], [1]] indices and per-group hyperparams."""
        p1 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.Adadelta([
            {"params": [p1], "lr": 0.4, "rho": 0.7},
            {"params": [p2], "eps": 1e-4},
        ])
        p1.grad = torch.ones_like(p1)
        p2.grad = torch.ones_like(p2)
        opt.step()
        seen = []
        opt.register_state_dict_post_hook(lambda o, sd: seen.append(sd) or None)
        opt.state_dict()
        packed = seen[0]
        self.assertEqual([g["params"] for g in packed["param_groups"]], [[0], [1]],
                         "each group's params must be packed as int indices")
        self.assertEqual(sorted(packed["state"].keys()), [0, 1],
                         "state must be keyed by int order indices across groups")
        g1, g2 = packed["param_groups"]
        self.assertEqual(g1["lr"], 0.4)
        self.assertEqual(g1["rho"], 0.7)
        self.assertEqual(g2["lr"], 1.0, "omitted lr must inherit Adadelta default 1.0")
        self.assertEqual(g2["eps"], 1e-4)
        # Both state slots carry the Adadelta key set after a step.
        for slot in (0, 1):
            self.assertEqual(sorted(packed["state"][slot].keys()),
                             ["acc_delta", "square_avg", "step"])

    # ------------------------------------------------------------------
    # Return-value semantics
    # ------------------------------------------------------------------

    def test_hook_return_dict_replaces_output(self):
        """A returned dict replaces the state_dict() output entirely."""
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
            lr=0.3)
        opt.register_state_dict_post_hook(lambda o, sd: {"replaced": True})
        out = opt.state_dict()
        self.assertEqual(out, {"replaced": True},
                         "the returned dict must replace the packed output")

    def test_hook_return_none_keeps_original(self):
        """Returning None keeps the packed dict as the output."""
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
            lr=0.3)
        opt.register_state_dict_post_hook(lambda o, sd: None)
        out = opt.state_dict()
        self.assertEqual(sorted(out.keys()), ["param_groups", "state"],
                         "returning None must keep the original packed dict")
        self.assertEqual(len(out["param_groups"]), 1)

    def test_hook_inplace_mutation_visible(self):
        """In-place edits to the packed dict (returning None) reach the output."""
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)],
            lr=0.3)

        def mutate(o, sd):
            sd["param_groups"][0]["lr"] = 99.0
            sd["touched"] = "yes"
            return None

        opt.register_state_dict_post_hook(mutate)
        out = opt.state_dict()
        self.assertEqual(out["param_groups"][0]["lr"], 99.0,
                         "in-place lr edit must be visible in the output")
        self.assertEqual(out.get("touched"), "yes",
                         "in-place key insertion must be visible in the output")

    # ------------------------------------------------------------------
    # Prepend ordering (two hooks, [B, A] with prepend vs [C, D] control)
    # ------------------------------------------------------------------

    def test_hook_prepend_ordering(self):
        """prepend=True fires the later-registered hook first: [B, A] vs [C, D]."""
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)])
        order = []
        opt.register_state_dict_post_hook(lambda o, sd: order.append("A") or None)
        # prepend=True: B is placed before the already-registered A.
        opt.register_state_dict_post_hook(lambda o, sd: order.append("B") or None,
                                          prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must fire the later-registered hook first")
        # Control: default (prepend=False) keeps registration order [C, D].
        opt2 = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)])
        order2 = []
        opt2.register_state_dict_post_hook(lambda o, sd: order2.append("C") or None)
        opt2.register_state_dict_post_hook(lambda o, sd: order2.append("D") or None)
        opt2.state_dict()
        self.assertEqual(order2, ["C", "D"],
                         "default registration order must be preserved")

    # ------------------------------------------------------------------
    # Removal semantics
    # ------------------------------------------------------------------

    def test_hook_removed_stops_firing(self):
        """After handle.remove() the hook no longer fires."""
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)])
        calls = []
        handle = opt.register_state_dict_post_hook(lambda o, sd: calls.append(1))
        opt.state_dict()
        self.assertEqual(len(calls), 1, "hook must fire before removal")
        handle.remove()
        opt.state_dict()
        opt.state_dict()
        self.assertEqual(len(calls), 1,
                         "removed hook must not fire on subsequent state_dict calls")

    def test_hook_remove_one_of_two(self):
        """Removing one hook keeps the other one firing."""
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)])
        calls = []
        h_a = opt.register_state_dict_post_hook(lambda o, sd: calls.append("A") or None)
        opt.register_state_dict_post_hook(lambda o, sd: calls.append("B") or None)
        opt.state_dict()
        self.assertEqual(calls, ["A", "B"])
        h_a.remove()
        opt.state_dict()
        self.assertEqual(calls, ["A", "B", "B"],
                         "only B must keep firing after A is removed")

    # ------------------------------------------------------------------
    # Non-callable hook path
    # ------------------------------------------------------------------

    def test_hook_non_callable_raises_on_state_dict(self):
        """Registering a non-callable does not raise; state_dict() then raises TypeError.

        Measured contract: registration itself is unchecked, the failure is
        deferred to the first state_dict() call that tries to invoke the hook.
        """
        opt = torch.optim.Adadelta(
            [torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)])
        handle = opt.register_state_dict_post_hook("not-callable")
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        with self.assertRaises(TypeError):
            opt.state_dict()
        # Cleaning up keeps later state_dict calls of THIS instance working.
        handle.remove()
        out = opt.state_dict()
        self.assertEqual(sorted(out.keys()), ["param_groups", "state"])


if __name__ == "__main__":
    run_tests()
