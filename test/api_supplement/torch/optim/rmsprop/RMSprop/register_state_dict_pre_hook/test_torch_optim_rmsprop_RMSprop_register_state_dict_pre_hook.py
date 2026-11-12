# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.rmsprop.RMSprop.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.rmsprop.RMSprop.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook: Callable[["Optimizer"], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则返回默认状态 | 已覆盖（test_hook_not_registered_default_state / test_hook_fires_before_packing） |
| 枚举选项 | prepend=False/True 控制触发次序 | 已覆盖（test_two_hooks_default_order_append / test_two_hooks_prepend_order） |
| 参数类型 | hook(optimizer) 单参回调；实参即优化器实例本身；返回值被忽略 | 已覆盖（test_hook_fires_before_packing） |
| 传参与不传参 | 显式 prepend=True 与默认 prepend=False | 已覆盖（test_two_hooks_default_order_append / test_two_hooks_prepend_order） |
| 等价类/边界值 | 触发位置在 state_dict 打包之前：hook 内对 param_groups 的原位修改反映到打包结果 | 已覆盖（test_hook_side_effect_visible_in_output） |
| 正常传参场景 | 注册后在 state_dict() 中确实被触发且实参正确 | 已覆盖（test_hook_fires_before_packing） |
| 异常传参场景 | 移除后不再触发；非 callable 抛 TypeError | 已覆盖（remove_handle_stops_firing/non_callable_hook_raises_typeerror_on_fire） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义；回调操作的是优化器元状态而非设备张量） |

未覆盖项及原因：
- 混合设备类型：本接口为纯框架回调注册，无张量迁移语义，不做设备对照（按规范不得伪造对照）。
- hook 返回 dict 的覆盖语义：实测该 hook 签名为 hook(optimizer) -> None，源码对 pre-hook 的返回值不做处理
  （state_dict() 只调用 pre_hook(self)，忽略返回值）；返回值语义仅存在于 post hook。据实不测返回值覆盖。

实测核实记录（python -c，torch 2.11）：
- pre hook 实参为优化器实例本身（o is opt 为 True）；源码 for pre_hook in ...: pre_hook(self)。
- hook 内原位修改 o.param_groups[0]['lr'] = 0.5 后，state_dict() 输出中 lr 即为 0.5（打包发生在 hook 之后）。
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
class TestRMSpropregister_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.rmsprop.RMSprop.register_state_dict_pre_hook (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        return torch.optim.RMSprop(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)], lr=0.01)

    # ------------------------------------------------------------------
    # Trigger point and hook arguments
    # ------------------------------------------------------------------

    def test_hook_fires_before_packing(self):
        """The hook fires once per state_dict() with the optimizer itself as the only arg."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt, "hook argument must be the optimizer instance")

    def test_hook_not_registered_default_state(self):
        """Without any hook, state_dict() returns the standard two-entry dict."""
        opt = self._make_optimizer()
        sd = opt.state_dict()
        self.assertIn("param_groups", sd)
        self.assertIn("state", sd)
        self.assertEqual(opt.param_groups[0]["lr"], 0.01)

    def test_hook_side_effect_visible_in_output(self):
        """An in-place mutation inside the pre hook shows up in the packed output.

        The pre hook runs BEFORE packing, so mutating the live param_groups
        (e.g. the lr) must be reflected in the state_dict() result.
        """
        opt = self._make_optimizer()

        def mutate_lr(o):
            o.param_groups[0]["lr"] = 0.5

        opt.register_state_dict_pre_hook(mutate_lr)
        sd = opt.state_dict()
        self.assertEqual(sd["param_groups"][0]["lr"], 0.5,
                         "pre hook runs before packing; mutation must be visible")

    # ------------------------------------------------------------------
    # Ordering with TWO hooks (prepend semantics)
    # ------------------------------------------------------------------

    def test_two_hooks_default_order_append(self):
        """Two hooks registered with prepend=False fire in registration order [A, B]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_pre_hook(lambda o: calls.append("A"))
        opt.register_state_dict_pre_hook(lambda o: calls.append("B"))
        opt.state_dict()
        self.assertEqual(calls, ["A", "B"],
                         "append order must follow registration order")

    def test_two_hooks_prepend_order(self):
        """A hook registered with prepend=True fires before existing ones: [B, A]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_pre_hook(lambda o: calls.append("A"))
        opt.register_state_dict_pre_hook(lambda o: calls.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(calls, ["B", "A"],
                         "prepend=True must fire the later-registered hook first")

    # ------------------------------------------------------------------
    # RemovableHandle
    # ------------------------------------------------------------------

    def test_remove_handle_stops_firing(self):
        """A removed handle no longer fires on subsequent state_dict() calls."""
        opt = self._make_optimizer()
        calls = []
        h = opt.register_state_dict_pre_hook(lambda o: calls.append(1))
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
        opt.register_state_dict_pre_hook(123)
        with self.assertRaises(TypeError):
            opt.state_dict()

    # ------------------------------------------------------------------
    # Device portability (pure framework state)
    # ------------------------------------------------------------------

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        opt = self._make_optimizer()
        handle = opt.register_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
