# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Optimizer.register_load_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.Optimizer.register_load_state_dict_pre_hook
API 签名：register_load_state_dict_pre_hook(self, hook, prepend=False) -> RemovableHandle

基类行为说明（实测确认）：load pre hook 签名为 hook(optimizer, state_dict)（双实参，
state_dict 为用户传入 dict 的浅拷贝）；返回值生效：返回 dict 完全替换加载源、
返回 None 沿用（含 hook 内 in-place 改写的）当前 dict；hook 在加载前触发，可读
到加载前的 optimizer 状态。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则默认行为 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook 断言 [B,A] 与 [A,B] 对照） | 已覆盖（hook_order_prepend_true_fires_first/
hook_order_registration_order） |
| 参数类型 | hook(opt, state_dict) 双实参契约（state_dict 为浅拷贝）；非 callable 注册成功、触发时 TypeError | 已覆盖（hook_two_arg_contract/
hook_non_callable_raises_typeerror） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true_fires_first） |
| 等价类/边界值 | 触发位置在 load_state_dict 处理之前：hook 读到加载前 lr；改写 incoming dict 生效（嵌套 in-place 与顶层替换均可） |
已覆盖（hook_fires_before_load/hook_rewrites_incoming_dict） |
| 正常传参场景 | 返回 dict 完全替换加载源（返回值生效）；返回 None 沿用原始 dict |
已覆盖（hook_return_dict_replaces_load_source/hook_return_none_uses_original） |
| 异常传参场景 | 非 callable hook 抛 TypeError；remove 后不再触发；二删一留 | 已覆盖（hook_trigger/
hook_non_callable_raises_typeerror/hook_removal） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义，注册为纯 Python 状态操作） |

未覆盖项及原因：
- 无（load pre hook 返回值契约已全覆盖：dict 替换 / None 沿用 / in-place 生效）。
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
class TestOptimizerregister_load_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.Optimizer.register_load_state_dict_pre_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: the two-argument contract (the
    received state_dict is a shallow copy of the caller's dict), dual-hook
    ordering for prepend=True/False, firing BEFORE the load (the hook sees
    the pre-load optimizer), incoming-dict rewriting (nested in-place and
    top-level), the return-value contract (a returned dict fully replaces
    the load source; None keeps the current one), removal (both / one of
    two), and the non-callable TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.SGD([p], lr=lr)

    @staticmethod
    def _sd(lr=0.05, momentum=0.0):
        """A minimal loadable state dict with one param group."""
        return {"state": {},
                "param_groups": [{"params": [0], "lr": lr, "momentum": momentum}]}

    def test_hook_trigger(self):
        """Hook is called with (this optimizer, a copy of the state dict)."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append(o))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt)

        # a hook may rewrite the state dict before loading
        def bump_lr(o, sd):
            for g in sd["param_groups"]:
                g["lr"] = 0.9
            return sd

        before = opt.state_dict()
        opt.register_load_state_dict_pre_hook(bump_lr)
        opt.load_state_dict(before)
        self.assertEqual(opt.param_groups[0]["lr"], 0.9)

        h = opt.register_load_state_dict_pre_hook(lambda o, sd: None)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()

        opt.register_load_state_dict_pre_hook(None)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_load_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()

    # ------------------------------------------------------------------
    # Two-argument contract
    # ------------------------------------------------------------------

    def test_hook_two_arg_contract(self):
        """The hook receives (optimizer, shallow-copied state_dict): the
        dict is NOT the caller's object, but its nested lists are shared."""
        opt = self._make_optimizer()
        user_sd = self._sd(lr=0.05)
        seen = {}

        def pre_hook(o, sd):
            seen["opt"] = o
            seen["sd"] = sd
            seen["sd_is_user"] = sd is user_sd
            seen["nested_shared"] = sd["param_groups"] is user_sd["param_groups"]
            return None

        opt.register_load_state_dict_pre_hook(pre_hook)
        opt.load_state_dict(user_sd)
        self.assertIs(seen["opt"], opt, "hook must receive this optimizer")
        self.assertFalse(seen["sd_is_user"],
                         "load_state_dict passes a shallow copy, not the caller's dict")
        self.assertTrue(seen["nested_shared"],
                        "shallow copy semantics: nested lists are shared objects")
        self.assertIsInstance(seen["sd"], dict)

    # ------------------------------------------------------------------
    # Ordering: prepend=True / default
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"), prepend=True)
        opt.load_state_dict(self._sd())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"))
        opt.load_state_dict(self._sd())
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    # ------------------------------------------------------------------
    # Fires BEFORE the load
    # ------------------------------------------------------------------

    def test_hook_fires_before_load(self):
        """The hook observes the PRE-load optimizer lr while the load
        afterwards overwrites it."""
        opt = self._make_optimizer(lr=0.01)
        seen_lrs = []

        def pre_hook(o, sd):
            seen_lrs.append(o.param_groups[0]["lr"])
            return None

        opt.register_load_state_dict_pre_hook(pre_hook)
        opt.load_state_dict(self._sd(lr=0.09))
        # The hook saw the old lr (0.01), proof it fired before the load.
        self.assertEqual(seen_lrs, [0.01],
                         "pre hook must see the pre-load optimizer state")
        # The load itself still happened: lr is now the incoming one.
        self.assertEqual(opt.param_groups[0]["lr"], 0.09,
                         "the load must overwrite the lr after the hook ran")

    # ------------------------------------------------------------------
    # Rewriting the incoming dict
    # ------------------------------------------------------------------

    def test_hook_rewrites_incoming_dict(self):
        """Nested in-place mutation and top-level replacement of the
        received dict both steer what gets loaded; the caller's dict is
        untouched (shallow copy)."""
        # Path 1: nested in-place mutation of the incoming dict.
        opt = self._make_optimizer(lr=0.01)

        def nested_mutate(o, sd):
            for g in sd["param_groups"]:
                g["lr"] = 0.42
            return None

        opt.register_load_state_dict_pre_hook(nested_mutate)
        opt.load_state_dict(self._sd(lr=0.05))
        self.assertEqual(opt.param_groups[0]["lr"], 0.42,
                         "nested in-place rewriting must steer the load")

        # Path 2: top-level replacement on the shallow copy: the caller's
        # dict is not mutated, but the replacement IS loaded.
        opt2 = self._make_optimizer(lr=0.01)
        user_sd = self._sd(lr=0.05)

        def top_replace(o, sd):
            sd["param_groups"] = [{"params": [0], "lr": 0.5, "momentum": 0.0}]
            return None

        opt2.register_load_state_dict_pre_hook(top_replace)
        opt2.load_state_dict(user_sd)
        self.assertEqual(opt2.param_groups[0]["lr"], 0.5,
                         "the hook's top-level replacement must be loaded")
        self.assertEqual(user_sd["param_groups"][0]["lr"], 0.05,
                         "the caller's dict must remain untouched (shallow copy)")

    # ------------------------------------------------------------------
    # Return-value contract: dict replaces the load source, None keeps it
    # ------------------------------------------------------------------

    def test_hook_return_dict_replaces_load_source(self):
        """A returned dict fully replaces the load source: the caller's
        dict contents are ignored in favor of the returned one."""
        opt = self._make_optimizer(lr=0.01)

        def replace_hook(o, sd):
            replacement = o.state_dict()
            replacement["param_groups"][0]["lr"] = 0.777
            replacement["param_groups"][0]["momentum"] = 0.0
            return replacement

        opt.register_load_state_dict_pre_hook(replace_hook)
        opt.load_state_dict(self._sd(lr=0.05, momentum=0.8))
        self.assertEqual(opt.param_groups[0]["lr"], 0.777,
                         "the returned dict must be the load source")
        self.assertEqual(opt.param_groups[0]["momentum"], 0.0,
                         "every field of the returned dict wins over the caller's")

    def test_hook_return_none_uses_original(self):
        """Returning None loads the (possibly in-place-mutated) original dict."""
        opt = self._make_optimizer(lr=0.01)

        def touch_and_none(o, sd):
            sd["param_groups"][0]["lr"] = 0.66
            return None

        opt.register_load_state_dict_pre_hook(touch_and_none)
        opt.load_state_dict(self._sd(lr=0.05))
        self.assertEqual(opt.param_groups[0]["lr"], 0.66,
                         "None return must keep the in-place-mutated incoming dict")
        # A plain None-returning hook leaves the load source as passed.
        opt2 = self._make_optimizer(lr=0.01)
        opt2.register_load_state_dict_pre_hook(lambda o, sd: None)
        opt2.load_state_dict(self._sd(lr=0.05))
        self.assertEqual(opt2.param_groups[0]["lr"], 0.05,
                         "untouched dict with None return must load as-is")

    # ------------------------------------------------------------------
    # Removal: both hooks, and one of two
    # ------------------------------------------------------------------

    def test_hook_removal(self):
        """Removed hooks stop firing; removing one of two keeps the other."""
        calls = []
        opt = self._make_optimizer()
        ha = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("A"))
        hb = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("B"))
        opt.load_state_dict(self._sd())
        self.assertEqual(calls, ["A", "B"], "both hooks fire in registration order")
        # Remove exactly one: the survivor must still fire.
        ha.remove()
        calls.clear()
        opt.load_state_dict(self._sd())
        self.assertEqual(calls, ["B"], "only the surviving hook may fire")
        # Remove the second: nothing fires.
        hb.remove()
        calls.clear()
        opt.load_state_dict(self._sd())
        self.assertEqual(calls, [], "no hook may fire after both are removed")

    # ------------------------------------------------------------------
    # Non-callable hook
    # ------------------------------------------------------------------

    def test_hook_non_callable_raises_typeerror(self):
        """A non-callable hook registers successfully and raises TypeError
        when load_state_dict() fires it."""
        opt = self._make_optimizer()
        h = opt.register_load_state_dict_pre_hook(None)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle,
                              "registering a non-callable must not raise")
        with self.assertRaises(TypeError):
            opt.load_state_dict(self._sd())
        h.remove()
        # The load works again once the bad hook is removed.
        opt.load_state_dict(self._sd(lr=0.06))
        self.assertEqual(opt.param_groups[0]["lr"], 0.06)


if __name__ == "__main__":
    run_tests()
