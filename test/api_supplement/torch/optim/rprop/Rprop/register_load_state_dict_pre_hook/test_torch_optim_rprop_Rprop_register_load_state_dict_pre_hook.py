# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.rprop.Rprop.register_load_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.rprop.Rprop.register_load_state_dict_pre_hook
API 签名：register_load_state_dict_pre_hook(self, hook: Callable[[Optimizer, StateDict], StateDict | None], prepend=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则正常加载 | 已覆盖（test_hook_not_registered_default_load / test_hook_fires_with_exact_args） |
| 枚举选项 | prepend=False/True 控制触发次序 | 已覆盖（test_two_hooks_default_order_append / test_two_hooks_prepend_order） |
| 参数类型 | hook(optimizer, state_dict) 双参回调；返回 dict 可替换待加载内容 | 已覆盖（hook_fires_with_exact_args/
hook_return_replaces_state_dict） |
| 传参与不传参 | 显式 prepend=True 与默认 prepend=False；hook 返回 None 走原 dict | 已覆盖（two_hooks_default_order_append/
two_hooks_prepend_order/hook_return_replaces_state_dict） |
| 等价类/边界值 | 触发位置在加载校验之前：hook 可改写待载入 dict（如 lr）且改写生效 | 已覆盖（hook_can_rewrite_incoming_dict/
hook_fires_before_validation_error） |
| 正常传参场景 | 注册后在 load_state_dict() 中确实被触发、实参正确、改写生效 | 已覆盖（test_hook_fires_with_exact_args / test_hook_can_rewrite_incoming_dict） |
| 异常传参场景 | 移除后不再触发；非 callable hook 注册成功、触发时抛 TypeError | 已覆盖（remove_handle_stops_firing/
non_callable_hook_raises_typeerror_on_fire） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义；回调操作的是优化器元状态而非设备张量） |

未覆盖项及原因：
- 混合设备类型：本接口为纯框架回调注册，无张量迁移语义，不做设备对照（按规范不得伪造对照）。

实测核实记录（python -c，torch 2.11）：
- hook 实参为 (optimizer, state_dict)：o is opt 为 True；sd 为 load_state_dict 内 state_dict.copy() 后的浅拷贝。
- hook 返回 dict 时该 dict 完全替换待加载内容（lr 改写生效）；返回 None 时沿用原 dict。
- 默认顺序注册 A、B 触发序为 [A, B]；B 以 prepend=True 注册时触发序为 [B, A]。
- 非 callable 注册时不抛错，在 load_state_dict() 触发 hook 时抛 TypeError。
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
class TestRpropregister_load_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.rprop.Rprop.register_load_state_dict_pre_hook (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        return torch.optim.Rprop(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)], lr=0.01)

    # ------------------------------------------------------------------
    # Trigger point and hook arguments
    # ------------------------------------------------------------------

    def test_hook_fires_with_exact_args(self):
        """The hook fires once per load_state_dict with (optimizer, state_dict)."""
        opt = self._make_optimizer()
        seen = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: seen.append((o, sd)))
        sd_in = opt.state_dict()
        opt.load_state_dict(sd_in)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], opt, "first hook arg must be the optimizer instance")
        # The dict passed to the hook is the caller's dict (shallow-copied),
        # i.e. it carries the same content the caller supplied.
        self.assertEqual(seen[0][1]["param_groups"][0]["lr"], 0.01)
        self.assertIsInstance(seen[0][1], dict)

    def test_hook_not_registered_default_load(self):
        """Without any hook, load_state_dict loads normally."""
        opt = self._make_optimizer()
        opt.load_state_dict(opt.state_dict())  # must not raise
        self.assertEqual(opt.param_groups[0]["lr"], 0.01)

    def test_hook_fires_before_validation_error(self):
        """The hook fires BEFORE the load validation (observable via group-count error)."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append(1))
        bad = {"state": {}, "param_groups": []}  # 0 groups vs 1 expected -> ValueError
        with self.assertRaises(ValueError):
            opt.load_state_dict(bad)
        # The hook already ran when the validation error was raised.
        self.assertEqual(len(calls), 1,
                         "pre hook must fire before load validation fails")

    # ------------------------------------------------------------------
    # Ordering with TWO hooks (prepend semantics)
    # ------------------------------------------------------------------

    def test_two_hooks_default_order_append(self):
        """Two hooks registered with prepend=False fire in registration order [A, B]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(calls, ["A", "B"],
                         "append order must follow registration order")

    def test_two_hooks_prepend_order(self):
        """A hook registered with prepend=True fires before existing ones: [B, A]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(calls, ["B", "A"],
                         "prepend=True must fire the later-registered hook first")

    # ------------------------------------------------------------------
    # State-dict rewriting semantics
    # ------------------------------------------------------------------

    def test_hook_can_rewrite_incoming_dict(self):
        """An in-place rewrite of the incoming dict in the hook takes effect."""
        opt = self._make_optimizer()

        def bump_lr(o, sd):
            for g in sd["param_groups"]:
                g["lr"] = 0.9
            return sd

        opt.register_load_state_dict_pre_hook(bump_lr)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(opt.param_groups[0]["lr"], 0.9,
                         "hook rewrite of the incoming lr must be loaded")

    def test_hook_return_replaces_state_dict(self):
        """A dict RETURNED by the hook fully replaces the incoming state dict."""
        opt = self._make_optimizer()

        def replace_lr(o, sd):
            new_sd = {"state": {}, "param_groups": [dict(g) for g in sd["param_groups"]]}
            for g in new_sd["param_groups"]:
                g["lr"] = 0.123
            return new_sd

        opt.register_load_state_dict_pre_hook(replace_lr)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(opt.param_groups[0]["lr"], 0.123,
                         "returned dict must be used as the loaded state dict")

    def test_hook_return_none_uses_original(self):
        """A hook returning None lets the original incoming dict load unchanged."""
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(lambda o, sd: None)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(opt.param_groups[0]["lr"], 0.01,
                         "None return must fall back to the original dict")

    # ------------------------------------------------------------------
    # RemovableHandle
    # ------------------------------------------------------------------

    def test_remove_handle_stops_firing(self):
        """A removed handle no longer fires on subsequent load_state_dict calls."""
        opt = self._make_optimizer()
        calls = []
        h = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append(1))
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.load_state_dict(opt.state_dict())
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    # ------------------------------------------------------------------
    # Non-callable hook
    # ------------------------------------------------------------------

    def test_non_callable_hook_raises_typeerror_on_fire(self):
        """A non-callable hook is accepted at registration but raises TypeError when fired."""
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(None)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    # ------------------------------------------------------------------
    # Device portability (pure framework state)
    # ------------------------------------------------------------------

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        opt = self._make_optimizer()
        handle = opt.register_load_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
