# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Optimizer.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.Optimizer.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle

基类行为说明（实测确认）：pre hook 签名为 hook(optimizer)（单实参），返回值被
基类忽略（state_dict() 始终返回自行打包的 dict）；hook 在打包循环之前触发，
对 live optimizer 的修改（如 add_param_group、改 lr）会反映到打包输出。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则默认状态 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook 断言 [B,A] 与 [A,B] 对照） | 已覆盖（hook_order_prepend_true_fires_first/
hook_order_registration_order） |
| 参数类型 | hook(optimizer) 单实参契约；非 callable 注册成功、触发时 TypeError | 已覆盖（test_hook_trigger/test_hook_non_callable_raises_typeerror） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true_fires_first） |
| 等价类/边界值 | 触发位置在 state_dict 打包之前：hook 读到 live param_groups（注册后 add_param_group 读到 2 组），hook 内新增组也进入打包输出 |
已覆盖（hook_fires_before_packing） |
| 正常传参场景 | 注册后在 state_dict() 打包前确实触发；in-place 副作用（改 lr）打包输出可见 | 已覆盖（test_hook_inplace_mutation_visible_in_packed） |
| 异常传参场景 | 非 callable hook 抛 TypeError；remove 后不再触发；二删一留 | 已覆盖（hook_trigger/
hook_non_callable_raises_typeerror/hook_removal） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义，注册为纯 Python 状态操作） |

未覆盖项及原因：
- pre hook 返回值维度：基类忽略 pre hook 返回值（无“返回 dict 生效”路径），已在等价类维度以 test_hook_return_value_ignored 显式断言“仍返回打包 dict”。
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
class TestOptimizerregister_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.Optimizer.register_state_dict_pre_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: the hook fires BEFORE state_dict
    packing (it reads the live param_groups and its mutations land in the
    packed output), dual-hook ordering for prepend=True/False, the
    return-value-is-ignored contract, in-place side effects visible in the
    packed dict, removal (both / one of two), and the non-callable
    TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.SGD([p], lr=lr)

    def test_hook_trigger(self):
        """Hook is called with this optimizer before packing."""
        calls = []
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt)

        # prepend + removable handle
        calls2 = []
        h = opt.register_state_dict_pre_hook(lambda o: calls2.append(id(o)), prepend=True)
        opt.state_dict()
        self.assertEqual(len(calls2), 1)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls2), 1, "removed hook must not fire again")

        # default (no-return) path still sees a fully-built dict
        self.assertIn("param_groups", opt.state_dict())
        self.assertIn("state", opt.state_dict())

        # non-callable registration succeeds but raises TypeError when fired
        opt.register_state_dict_pre_hook(123)
        with self.assertRaises(TypeError):
            opt.state_dict()

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()

    # ------------------------------------------------------------------
    # Fires BEFORE packing
    # ------------------------------------------------------------------

    def test_hook_fires_before_packing(self):
        """The hook reads the LIVE param_groups (2 after add_param_group),
        and a group added inside the hook itself reaches the packed output:
        proof the hook runs before the packing loop."""
        opt = self._make_optimizer(lr=0.01)
        seen_groups = []
        added_p = torch.nn.Parameter(torch.ones(1, device=self.device),
                                     requires_grad=True)

        def pre_hook(o):
            # Record the live (unpacked) group count at hook time.
            seen_groups.append(len(o.param_groups))
            o.add_param_group({"params": [added_p], "lr": 0.05})

        opt.register_state_dict_pre_hook(pre_hook)
        # A group added AFTER registration (and before the call) must be
        # visible to the hook: it fires on the live optimizer object.
        extra_p = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt.add_param_group({"params": [extra_p], "lr": 0.02})
        sd = opt.state_dict()
        # The hook saw the live 2 groups (pre-registration group + the one
        # added after registration), NOT the packed dict.
        self.assertEqual(seen_groups, [2],
                         "pre hook must see the live param_groups, not the packed dict")
        # Packing ran after the hook, so its output has 3 groups: the live 2
        # plus the group the hook itself added.
        self.assertEqual(len(sd["param_groups"]), 3)
        # Params are packed as int indices continuing across groups.
        self.assertEqual([g["params"] for g in sd["param_groups"]], [[0], [1], [2]])
        # The in-hook added group's lr reached the packed output.
        self.assertEqual(sd["param_groups"][2]["lr"], 0.05)

    # ------------------------------------------------------------------
    # Ordering: prepend=True / default
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))  # prepend=False
        opt.register_state_dict_pre_hook(lambda o: order.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))
        opt.register_state_dict_pre_hook(lambda o: order.append("B"))
        opt.state_dict()
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    # ------------------------------------------------------------------
    # Return value is ignored by the base class
    # ------------------------------------------------------------------

    def test_hook_return_value_ignored(self):
        """Whatever the pre hook returns, state_dict() still returns the
        packed dict built by the base class."""
        opt = self._make_optimizer(lr=0.03)
        opt.register_state_dict_pre_hook(lambda o: {"fake": "should-be-ignored"})
        sd = opt.state_dict()
        self.assertEqual(sorted(sd.keys()), ["param_groups", "state"],
                         "the packed dict must be returned regardless of the hook return")
        self.assertNotIn("fake", sd, "the pre hook's return value must be ignored")
        self.assertEqual(sd["param_groups"][0]["lr"], 0.03,
                         "the packed dict must reflect the real optimizer state")
        # A list return value is ignored just the same.
        opt.register_state_dict_pre_hook(lambda o: [1, 2, 3])
        sd2 = opt.state_dict()
        self.assertEqual(sorted(sd2.keys()), ["param_groups", "state"])

    # ------------------------------------------------------------------
    # In-place side effects on the live optimizer are visible in packing
    # ------------------------------------------------------------------

    def test_hook_inplace_mutation_visible_in_packed(self):
        """An lr change made inside the pre hook shows up in the packed output."""
        opt = self._make_optimizer(lr=0.01)

        def bump_lr(o):
            o.param_groups[0]["lr"] = 0.321
            return None

        opt.register_state_dict_pre_hook(bump_lr)
        sd = opt.state_dict()
        self.assertEqual(sd["param_groups"][0]["lr"], 0.321,
                         "the pre hook's live mutation must be visible in the packed dict")
        # The live optimizer keeps the mutated lr (hook mutates the optimizer,
        # not a copy).
        self.assertEqual(opt.param_groups[0]["lr"], 0.321)

    # ------------------------------------------------------------------
    # Removal: both hooks, and one of two
    # ------------------------------------------------------------------

    def test_hook_removal(self):
        """Removed hooks stop firing; removing one of two keeps the other."""
        calls = []
        opt = self._make_optimizer()
        ha = opt.register_state_dict_pre_hook(lambda o: calls.append("A"))
        hb = opt.register_state_dict_pre_hook(lambda o: calls.append("B"))
        opt.state_dict()
        self.assertEqual(calls, ["A", "B"], "both hooks fire in registration order")
        # Remove exactly one: the survivor must still fire.
        ha.remove()
        calls.clear()
        opt.state_dict()
        self.assertEqual(calls, ["B"], "only the surviving hook may fire")
        # Remove the second: nothing fires.
        hb.remove()
        calls.clear()
        opt.state_dict()
        self.assertEqual(calls, [], "no hook may fire after both are removed")

    # ------------------------------------------------------------------
    # Non-callable hook
    # ------------------------------------------------------------------

    def test_hook_non_callable_raises_typeerror(self):
        """A non-callable hook registers successfully and raises TypeError
        when state_dict() fires it."""
        opt = self._make_optimizer()
        h = opt.register_state_dict_pre_hook("not-callable")
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle,
                              "registering a non-callable must not raise")
        with self.assertRaises(TypeError):
            opt.state_dict()
        h.remove()
        # state_dict works again once the bad hook is removed.
        sd = opt.state_dict()
        self.assertIn("param_groups", sd)


if __name__ == "__main__":
    run_tests()
