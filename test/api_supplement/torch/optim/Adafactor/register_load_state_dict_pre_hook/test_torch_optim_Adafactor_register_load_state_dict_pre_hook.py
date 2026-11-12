# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Adafactor.register_load_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.Adafactor.register_load_state_dict_pre_hook
API 签名：register_load_state_dict_pre_hook(hook: Callable[[Optimizer, dict], dict|None], prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则不触发 | 已覆盖（_fires_with_arguments/_removed_does_not_fire） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook [A,B]/[B,A]） | 已覆盖（_order_default_registration/_order_prepend） |
| 参数类型 | hook(opt, state_dict) 双参回调；返回 RemovableHandle | 已覆盖（_fires_with_arguments/_removed_does_not_fire） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（_order_default_registration/_order_prepend） |
| 等价类/边界值 | 载入前触发；改写/替换/沿用 dict 生效 | 已覆盖（_can_rewrite_state_dict/_return_value_replaces_dict/_none_return_keeps_dict） |
| 正常传参场景 | 注册后在 load_state_dict() 中确实被触发且实参为 (优化器, 待载入 dict) | 已覆盖（_fires_with_arguments） |
| 异常传参场景 | 非 callable hook：注册成功、load 时抛 TypeError | 已覆盖（_non_callable_raises_on_load） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义；state 观测用例在当前设备上执行） |

未覆盖项及原因：
- 无（hook 为纯 Python 回调注册，无张量数值语义；实测确认注册非 callable 不会立刻报错，
  TypeError 在 load_state_dict 调用 hook 时抛出，消息为 "'NoneType' object is not callable"）。
"""
import copy
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
class TestAdafactorregister_load_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.Adafactor.register_load_state_dict_pre_hook (NPU/CUDA portable).

    Adafactor inherits this hook registry from Optimizer, but the file still
    exercises the full contract independently on an Adafactor instance:
    argument identity (optimizer, mutable state dict), the pre-load firing
    point (the dict can still be rewritten and the rewrite takes effect), the
    return-value replacement semantics, dual-hook ordering for both prepend
    values, removal and non-callable error timing.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adafactor([p], lr=0.01)

    def _stepped_isolated_state_dict(self, opt):
        """Step once and return a deepcopy'd state_dict (live-reference safe)."""
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.full_like(p, 0.5)
        opt.step()
        return copy.deepcopy(opt.state_dict())

    def test_pre_hook_fires_with_arguments(self):
        """Hook receives (optimizer, state_dict copy) with the optimizer identity.

        Measured contract: load_state_dict shallow-copies the input dict before
        invoking pre hooks (consistent with the module API), so the dict the
        hook sees is value-equal to the caller's dict but not the same object.
        In-place edits inside the hook still target the payload being loaded.
        """
        opt = self._make_optimizer()
        seen = []
        opt.register_load_state_dict_pre_hook(
            lambda o, sd: seen.append((o, sd)) or None)
        sd_in = opt.state_dict()
        opt.load_state_dict(sd_in)
        self.assertEqual(len(seen), 1, "hook must fire exactly once per load")
        # First argument must be the optimizer instance itself.
        self.assertIs(seen[0][0], opt)
        # Second argument: the shallow copy load_state_dict made of sd_in
        # (source: state_dict.copy() before hook dispatch).
        self.assertIsNot(seen[0][1], sd_in)
        self.assertEqual(seen[0][1], sd_in,
                         "hook payload must be a value-equal copy of the input")

    def test_pre_hook_can_rewrite_state_dict(self):
        """Pre-load timing: mutating the dict inside the hook takes effect."""
        opt = self._make_optimizer()
        sd = self._stepped_isolated_state_dict(opt)

        def bump_lr(o, state_dict):
            for group in state_dict["param_groups"]:
                group["lr"] = 0.9
            return None  # in-place rewrite, no replacement dict

        opt.register_load_state_dict_pre_hook(bump_lr)
        opt.load_state_dict(sd)
        # The hook's rewrite of lr survived into the loaded param group.
        self.assertEqual(opt.param_groups[0]["lr"], 0.9,
                         "pre hook must run BEFORE the dict is applied")

    def test_pre_hook_return_value_replaces_dict(self):
        """Returning a dict from the hook replaces the dict being loaded."""
        opt = self._make_optimizer()
        sd = self._stepped_isolated_state_dict(opt)
        # Build a replaced dict with a recognizable lr.
        replaced = copy.deepcopy(sd)
        for group in replaced["param_groups"]:
            group["lr"] = 0.3
        opt.register_load_state_dict_pre_hook(lambda o, s: replaced)
        opt.load_state_dict(sd)
        # The loaded optimizer must reflect the REPLACED dict (lr=0.3), not the
        # original sd (lr=0.01): the return value swaps the load payload.
        self.assertEqual(opt.param_groups[0]["lr"], 0.3)

    def test_pre_hook_none_return_keeps_dict(self):
        """Returning None keeps the (possibly in-place mutated) original dict."""
        opt = self._make_optimizer()
        sd = copy.deepcopy(self._stepped_isolated_state_dict(opt))
        # Sanity: after one step the variance mean is 0.25 for grad=0.5.
        opt.register_load_state_dict_pre_hook(lambda o, s: None)
        opt.load_state_dict(sd)
        p = opt.param_groups[0]["params"][0]
        # The untouched payload is applied as-is.
        self.assertEqual(opt.param_groups[0]["lr"], 0.01)
        torch.testing.assert_close(opt.state[p]["variance"],
                                   sd["state"][0]["variance"])

    def test_pre_hook_order_default_registration(self):
        """prepend=False (default): hooks fire in registration order [A, B]."""
        opt = self._make_optimizer()
        order = []
        opt.register_load_state_dict_pre_hook(
            lambda o, sd: order.append("A") or None)
        opt.register_load_state_dict_pre_hook(
            lambda o, sd: order.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["A", "B"])

    def test_pre_hook_order_prepend(self):
        """prepend=True: the later-registered hook fires first, giving [B, A]."""
        opt = self._make_optimizer()
        order = []
        opt.register_load_state_dict_pre_hook(
            lambda o, sd: order.append("A") or None)
        opt.register_load_state_dict_pre_hook(
            lambda o, sd: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"])

    def test_pre_hook_removed_does_not_fire(self):
        """After handle.remove(), the hook is not invoked by further loads."""
        opt = self._make_optimizer()
        calls = []
        handle = opt.register_load_state_dict_pre_hook(
            lambda o, sd: calls.append(o) or None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        handle.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_pre_hook_non_callable_raises_on_load(self):
        """Registering a non-callable succeeds; TypeError surfaces at load time."""
        opt = self._make_optimizer()
        # Verified behavior: registration itself does not validate callability
        # and returns a valid handle; the error is raised when load_state_dict
        # tries to invoke the hook.
        handle = opt.register_load_state_dict_pre_hook(None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        with self.assertRaisesRegex(TypeError, "object is not callable"):
            opt.load_state_dict(opt.state_dict())
        handle.remove()


if __name__ == "__main__":
    run_tests()
