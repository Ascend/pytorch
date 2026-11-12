# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.sparse_adam.SparseAdam.register_load_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.sparse_adam.SparseAdam.register_load_state_dict_pre_hook
API 签名：register_load_state_dict_pre_hook(self, hook: Callable[[Optimizer, StateDict], StateDict | None], prepend=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则正常加载 | 已覆盖（test_hook_trigger/test_hook_no_registration_no_effect） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook [B,A] / 注册顺序 [A,B]） | 已覆盖（test_hook_order_prepend_true/test_hook_order_append） |
| 参数类型 | hook(optimizer, state_dict) 双实参契约（state_dict 为浅拷贝） | 已覆盖（test_hook_trigger/test_hook_receives_shallow_copy） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true） |
| 等价类/边界值 | 触发时点在 load 之前：hook 观察到旧 lr，返回 dict 生效后加载新值；in-place 修改生效；返回 dict 完全替换 | 已覆盖（hook_fires_before_load/
hook_inplace_edit_effective/hook_return_replaces_dict_to_load） |
| 正常传参场景 | 注册后 load_state_dict() 中触发；加载 NPU state 的 moments 张量落在 NPU 设备；RemovableHandle 移除后不再触发；二删一留 |
已覆盖（load_npu_state_via_hook/hook_remove_stops_firing/remove_one_of_two_keeps_other） |
| 异常传参场景 | 非 callable 注册不抛、触发时 TypeError；None 同理 | 已覆盖（test_non_callable_hook_raises_on_fire/test_none_hook_raises_on_fire） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 混合设备类型：register_load_state_dict_pre_hook 是纯 Python 回调注册，回调实参为 optimizer 与
  state_dict 浅拷贝，无张量跨设备迁移路径；加载后 state 张量设备归属由
  test_load_npu_state_via_hook 验证落在参数所在设备。
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
class TestSparseAdamregister_load_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.sparse_adam.SparseAdam.register_load_state_dict_pre_hook (NPU/CUDA portable).

    Covers: the (optimizer, state_dict) argument contract over a shallow
    copy, firing BEFORE the load (the hook sees the pre-load optimizer
    state; its edits and returns take effect), return-value replacement of
    the dict to load, double-hook ordering for prepend=True ([B, A]) and
    the default append order ([A, B]), handle removal, remove-one-keep-
    other, and the TypeError raised when a non-callable registration fires.

    Verified contract (upstream optimizer.load_state_dict): the state_dict
    handed to the hook is a shallow copy of the user's dict; a non-None
    hook return replaces the dict being loaded; in-place mutations of the
    copied dict take effect; the optimizer's own param_groups are updated
    only after all pre hooks have run.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_opt(self):
        p = torch.nn.Parameter(torch.randn(4, 3, device=self.device), requires_grad=True)
        return torch.optim.SparseAdam([p], lr=0.05)

    def _seeded_opt(self):
        """Optimizer with a non-empty state on the current device."""
        p = torch.nn.Parameter(torch.randn(4, 3, device=self.device), requires_grad=True)
        opt = torch.optim.SparseAdam([p], lr=0.05)
        opt.state[p] = {
            "step": 2,
            "exp_avg": torch.full_like(p, 0.1),
            "exp_avg_sq": torch.full_like(p, 0.2),
        }
        return opt

    # ------------------------------------------------------------------
    # Basic firing and argument contract
    # ------------------------------------------------------------------

    def test_hook_trigger(self):
        """Hook fires once per load_state_dict() with (optimizer, dict)."""
        opt = self._make_opt()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append((o, sd)))
        sd = opt.state_dict()
        opt.load_state_dict(sd)
        self.assertEqual(len(calls), 1,
                         "hook must fire exactly once per load_state_dict()")
        self.assertIs(calls[0][0], opt, "the first hook argument must be the optimizer")
        self.assertIsInstance(calls[0][1], dict)

    def test_hook_receives_shallow_copy(self):
        """The hook receives a top-level shallow copy: replacing whole keys
        never touches the caller's dict, while nested lists are shared."""
        opt = self._make_opt()
        sd = opt.state_dict()
        original_groups = sd["param_groups"]
        seen = {}

        def hook(o, s):
            seen["groups_same_object"] = s["param_groups"] is original_groups
            # Top-level replacement: only affects the dict being loaded.
            s["param_groups"] = [dict(original_groups[0], lr=0.99)]
            return None

        opt.register_load_state_dict_pre_hook(hook)
        opt.load_state_dict(sd)
        # The nested list is shared between the copy and the caller's dict
        # (verified shallow-copy semantics).
        self.assertTrue(seen["groups_same_object"],
                        "the copy shares nested objects with the caller's dict")
        self.assertIs(sd["param_groups"], original_groups,
                      "the caller's top-level keys must stay unmodified (shallow copy)")
        self.assertEqual(sd["param_groups"][0]["lr"], 0.05,
                         "the caller's nested lr must stay unmodified")
        self.assertEqual(opt.param_groups[0]["lr"], 0.99,
                         "the hook's replacement must be the dict actually loaded")

    def test_hook_no_registration_no_effect(self):
        """Without any registration load_state_dict() works normally."""
        opt = self._seeded_opt()
        sd = opt.state_dict()
        opt2 = self._make_opt()
        opt2.load_state_dict(sd)
        self.assertEqual(opt2.param_groups[0]["lr"], 0.05)

    # ------------------------------------------------------------------
    # Fires BEFORE the load: pre-load state is visible
    # ------------------------------------------------------------------

    def test_hook_fires_before_load(self):
        """The hook observes the optimizer's pre-load lr; the load then applies the new one."""
        opt = self._make_opt()
        captured = {}

        def hook(o, s):
            captured["lr_at_hook"] = o.param_groups[0]["lr"]

        opt.register_load_state_dict_pre_hook(hook)
        sd = opt.state_dict()
        sd["param_groups"][0]["lr"] = 0.42
        opt.load_state_dict(sd)
        self.assertEqual(captured["lr_at_hook"], 0.05,
                         "the pre hook must run before the load updates the groups")
        self.assertEqual(opt.param_groups[0]["lr"], 0.42,
                         "the loaded lr must be applied after the hook ran")

    def test_hook_inplace_edit_effective(self):
        """An in-place lr edit inside the hook wins over the dict being loaded."""
        opt = self._make_opt()

        def bump_lr(o, sd):
            for g in sd["param_groups"]:
                g["lr"] = 0.9
            return None

        sd = opt.state_dict()
        opt.register_load_state_dict_pre_hook(bump_lr)
        opt.load_state_dict(sd)
        self.assertEqual(opt.param_groups[0]["lr"], 0.9,
                         "in-place edits made by the pre hook must be loaded")

    def test_hook_return_replaces_dict_to_load(self):
        """A dict returned by the hook fully replaces the dict being loaded."""
        opt = self._make_opt()
        sd = opt.state_dict()

        def replace(o, s):
            s = dict(s)
            s["param_groups"] = [{
                "params": [0],
                "lr": 0.77,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "maximize": False,
            }]
            s["state"] = {0: {}}
            return s

        opt.register_load_state_dict_pre_hook(replace)
        opt.load_state_dict(sd)
        self.assertEqual(opt.param_groups[0]["lr"], 0.77,
                         "the hook's returned dict must be the one loaded")

    # ------------------------------------------------------------------
    # Real state loads through the hook on the device
    # ------------------------------------------------------------------

    def test_load_npu_state_via_hook(self):
        """Loading device moments through a pre hook lands them on the param's device."""
        if self.device_name not in ('npu', 'cuda'):
            self.skipTest('no accel device')
        src = self._seeded_opt()
        sd = src.state_dict()
        fired = []
        opt = self._make_opt()
        opt.register_load_state_dict_pre_hook(lambda o, s: fired.append(1))
        opt.load_state_dict(sd)
        self.assertEqual(len(fired), 1)
        p = opt.param_groups[0]["params"][0]
        self.assertEqual(opt.state[p]["step"], 2, "the step count must be loaded")
        self.assertEqual(opt.state[p]["exp_avg"].device.type, self.device_name,
                         "loaded moments must live on the param's device")
        torch.testing.assert_close(opt.state[p]["exp_avg"],
                                   torch.full_like(p, 0.1))
        torch.testing.assert_close(opt.state[p]["exp_avg_sq"],
                                   torch.full_like(p, 0.2))

    # ------------------------------------------------------------------
    # Ordering: prepend=True / default append, two hooks each
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true(self):
        """With prepend=True the later-registered hook fires first ([B, A])."""
        opt = self._make_opt()
        order = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the new hook to the front")

    def test_hook_order_append(self):
        """Default registration keeps the registration order ([A, B])."""
        opt = self._make_opt()
        order = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["A", "B"],
                         "without prepend the firing order is the registration order")

    # ------------------------------------------------------------------
    # Removal semantics
    # ------------------------------------------------------------------

    def test_hook_remove_stops_firing(self):
        """handle.remove() stops the hook; a double remove is a no-op."""
        opt = self._make_opt()
        calls = []
        h = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append(1))
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        h.remove()
        h.remove()  # second remove must not raise
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1, "a removed hook must not fire again")

    def test_remove_one_of_two_keeps_other(self):
        """Removing one of two hooks leaves the other one firing."""
        opt = self._make_opt()
        calls = []
        h1 = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("one"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("two"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(calls, ["one", "two"])
        h1.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(calls, ["one", "two", "two"],
                         "only the removed hook disappears; the survivor keeps firing")

    # ------------------------------------------------------------------
    # Non-callable registrations
    # ------------------------------------------------------------------

    def test_non_callable_hook_raises_on_fire(self):
        """Registering a non-callable succeeds; firing it raises TypeError."""
        opt = self._make_opt()
        opt.register_load_state_dict_pre_hook(3.14)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    def test_none_hook_raises_on_fire(self):
        """Registering None succeeds; firing it raises TypeError."""
        opt = self._make_opt()
        opt.register_load_state_dict_pre_hook(None)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    # ------------------------------------------------------------------
    # Device-independence sanity
    # ------------------------------------------------------------------

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_opt()
        handle = opt.register_load_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
