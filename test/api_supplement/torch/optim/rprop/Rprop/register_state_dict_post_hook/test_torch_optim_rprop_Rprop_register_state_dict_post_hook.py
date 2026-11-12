# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.rprop.Rprop.register_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.rprop.Rprop.register_state_dict_post_hook
API 签名：register_state_dict_post_hook(self, hook: Callable[[Optimizer, StateDict], StateDict | None], prepend=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则返回默认状态 | 已覆盖（test_hook_not_registered_default_state / test_hook_fires_after_packing） |
| 枚举选项 | prepend=False/True 控制触发次序 | 已覆盖（test_two_hooks_default_order_append / test_two_hooks_prepend_order） |
| 参数类型 | hook(optimizer, state_dict) 双参回调；返回 dict 可替换输出 | 已覆盖（test_hook_fires_after_packing / test_hook_return_replaces_output） |
| 传参与不传参 | 显式 prepend=True 与默认 prepend=False；hook 返回 None 保留原输出 | 已覆盖（two_hooks_default_order_append/
two_hooks_prepend_order/hook_return_none_keeps_original） |
| 等价类/边界值 | 触发位置在 state_dict 打包之后：hook 收到的即最终待返回 dict | 已覆盖（test_hook_fires_after_packing / test_hook_inplace_mutation_visible） |
| 正常传参场景 | 注册后在 state_dict() 返回值生成后确实触发、实参正确 | 已覆盖（test_hook_fires_after_packing / test_hook_inplace_mutation_visible） |
| 异常传参场景 | 移除后不再触发；非 callable hook 注册成功、触发时抛 TypeError | 已覆盖（remove_handle_stops_firing/
non_callable_hook_raises_typeerror_on_fire） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义；回调操作的是优化器元状态而非设备张量） |

未覆盖项及原因：
- 混合设备类型：本接口为纯框架回调注册，无张量迁移语义，不做设备对照（按规范不得伪造对照）。

实测核实记录（python -c，torch 2.11）：
- hook 实参为 (optimizer, state_dict)：o is opt 为 True；sd 为已打包的完整 state dict。
- hook 原位修改 sd（如注入键）会反映到 state_dict() 返回值；返回新 dict 时完全替换返回值（返回 {'custom': 1} 实测成立）。
- 默认顺序注册 A、B 触发序为 [A, B]；B 以 prepend=True 注册时触发序为 [B, A]。
- 非 callable 注册时不抛错，在 state_dict() 触发 hook 时抛 TypeError。
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
class TestRpropregister_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.rprop.Rprop.register_state_dict_post_hook (NPU/CUDA portable)."""

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

    def test_hook_fires_after_packing(self):
        """The hook fires once per state_dict() with (optimizer, packed state_dict)."""
        opt = self._make_optimizer()
        seen = []
        opt.register_state_dict_post_hook(lambda o, sd: seen.append((o, sd)))
        sd = opt.state_dict()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], opt, "first hook arg must be the optimizer instance")
        # The hook receives the fully packed dict that is about to be returned.
        self.assertIn("param_groups", seen[0][1])
        self.assertIn("state", seen[0][1])
        self.assertEqual(seen[0][1]["param_groups"][0]["lr"], 0.01)

    def test_hook_not_registered_default_state(self):
        """Without any hook, state_dict() returns the standard two-entry dict."""
        opt = self._make_optimizer()
        sd = opt.state_dict()
        self.assertIn("param_groups", sd)
        self.assertIn("state", sd)

    def test_hook_inplace_mutation_visible(self):
        """An in-place mutation of the packed dict inside the hook is returned."""
        opt = self._make_optimizer()

        def inject(o, sd):
            sd["extra"] = 42
            return sd

        opt.register_state_dict_post_hook(inject)
        out = opt.state_dict()
        self.assertEqual(out.get("extra"), 42,
                         "in-place mutation must be visible in the returned dict")

    # ------------------------------------------------------------------
    # Ordering with TWO hooks (prepend semantics)
    # ------------------------------------------------------------------

    def test_two_hooks_default_order_append(self):
        """Two hooks registered with prepend=False fire in registration order [A, B]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_post_hook(lambda o, sd: calls.append("A"))
        opt.register_state_dict_post_hook(lambda o, sd: calls.append("B"))
        opt.state_dict()
        self.assertEqual(calls, ["A", "B"],
                         "append order must follow registration order")

    def test_two_hooks_prepend_order(self):
        """A hook registered with prepend=True fires before existing ones: [B, A]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_post_hook(lambda o, sd: calls.append("A"))
        opt.register_state_dict_post_hook(lambda o, sd: calls.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(calls, ["B", "A"],
                         "prepend=True must fire the later-registered hook first")

    # ------------------------------------------------------------------
    # Return-value semantics
    # ------------------------------------------------------------------

    def test_hook_return_replaces_output(self):
        """A dict RETURNED by the hook fully replaces the state_dict() output."""
        opt = self._make_optimizer()
        opt.register_state_dict_post_hook(lambda o, sd: {"custom": 1})
        self.assertEqual(opt.state_dict(), {"custom": 1},
                         "returned dict must replace the packed output entirely")

    def test_hook_return_none_keeps_original(self):
        """A hook returning None keeps the packed dict as the output."""
        opt = self._make_optimizer()
        opt.register_state_dict_post_hook(lambda o, sd: None)
        sd = opt.state_dict()
        self.assertIn("param_groups", sd)
        self.assertIn("state", sd)

    # ------------------------------------------------------------------
    # RemovableHandle
    # ------------------------------------------------------------------

    def test_remove_handle_stops_firing(self):
        """A removed handle no longer fires on subsequent state_dict() calls."""
        opt = self._make_optimizer()
        calls = []
        h = opt.register_state_dict_post_hook(lambda o, sd: calls.append(1))
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.state_dict()
        opt.state_dict()
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    # ------------------------------------------------------------------
    # Non-callable hook
    # ------------------------------------------------------------------

    def test_non_callable_hook_raises_typeerror_on_fire(self):
        """A non-callable hook is accepted at registration but raises TypeError when fired."""
        opt = self._make_optimizer()
        opt.register_state_dict_post_hook("not-callable")
        with self.assertRaises(TypeError):
            opt.state_dict()

    # ------------------------------------------------------------------
    # Device portability (pure framework state)
    # ------------------------------------------------------------------

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        opt = self._make_optimizer()
        handle = opt.register_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
