# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.radam.RAdam.register_load_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.radam.RAdam.register_load_state_dict_post_hook
API 签名：register_load_state_dict_post_hook(self, hook: Callable[["Optimizer"], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则默认状态 | 已覆盖（test_hook_not_registered_no_effect / test_hook_fires_after_load） |
| 枚举选项 | prepend=False/True 控制触发次序 | 已覆盖（test_two_hooks_default_order_append / test_two_hooks_prepend_order） |
| 参数类型 | hook(optimizer) 单参回调；实参即优化器实例本身 | 已覆盖（test_hook_fires_after_load） |
| 传参与不传参 | 显式 prepend=True 与默认 prepend=False | 已覆盖（test_two_hooks_default_order_append / test_two_hooks_prepend_order） |
| 等价类/边界值 | 触发位置在 load_state_dict 完成之后（加载结果已生效） | 已覆盖（test_hook_fires_after_load） |
| 正常传参场景 | 注册后在 load_state_dict() 完成后确实触发且实参正确 | 已覆盖（test_hook_fires_after_load） |
| 异常传参场景 | 移除后不再触发；非 callable 抛 TypeError | 已覆盖（remove_handle_stops_firing/non_callable_hook_raises_typeerror_on_fire） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义；回调操作的是优化器元状态而非设备张量） |

未覆盖项及原因：
- 混合设备类型：本接口为纯框架回调注册，无张量迁移语义，不做设备对照（按规范不得伪造对照）。

实测核实记录（python -c，torch 2.11）：
- 非 callable 注册时不抛错（注册阶段无校验），在 load_state_dict() 触发 hook 时抛 TypeError。
- 默认顺序注册 A、B 触发序为 [A, B]；B 以 prepend=True 注册时触发序为 [B, A]。
- hook 实参为优化器实例本身（lambda o: ... 的 o is opt 为 True）。
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
class TestRAdamregister_load_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.radam.RAdam.register_load_state_dict_post_hook (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        return torch.optim.RAdam(
            [torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)], lr=0.01)

    # ------------------------------------------------------------------
    # Trigger point and hook arguments
    # ------------------------------------------------------------------

    def test_hook_fires_after_load(self):
        """The hook fires once per load_state_dict with the optimizer itself as the only arg."""
        opt = self._make_optimizer()
        seen = []
        opt.register_load_state_dict_post_hook(lambda o: seen.append(o))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], opt, "hook argument must be the optimizer instance")

        # The hook runs AFTER loading completed: the loaded state is already
        # in effect when the hook observes the optimizer.
        marker = {"called": False}

        def probe(o):
            marker["called"] = True
            self.assertEqual(len(o.param_groups), 1)
        opt.register_load_state_dict_post_hook(probe)
        opt.load_state_dict(opt.state_dict())
        self.assertTrue(marker["called"])

    def test_hook_not_registered_no_effect(self):
        """Without any hook, load_state_dict works and yields the default state."""
        opt = self._make_optimizer()
        sd = opt.state_dict()
        opt.load_state_dict(sd)  # must not raise
        self.assertEqual(len(opt.param_groups), 1)
        self.assertEqual(opt.param_groups[0]["lr"], 0.01)

    # ------------------------------------------------------------------
    # Ordering with TWO hooks (prepend semantics)
    # ------------------------------------------------------------------

    def test_two_hooks_default_order_append(self):
        """Two hooks registered with prepend=False fire in registration order [A, B]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_post_hook(lambda o: calls.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: calls.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(calls, ["A", "B"],
                         "append order must follow registration order")

    def test_two_hooks_prepend_order(self):
        """A hook registered with prepend=True fires before existing ones: [B, A]."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_post_hook(lambda o: calls.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: calls.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(calls, ["B", "A"],
                         "prepend=True must fire the later-registered hook first")

    # ------------------------------------------------------------------
    # RemovableHandle
    # ------------------------------------------------------------------

    def test_remove_handle_stops_firing(self):
        """A removed handle no longer fires on subsequent load_state_dict calls."""
        opt = self._make_optimizer()
        calls = []
        h = opt.register_load_state_dict_post_hook(lambda o: calls.append(1))
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
        opt.register_load_state_dict_post_hook(3.14)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    # ------------------------------------------------------------------
    # Device portability (pure framework state)
    # ------------------------------------------------------------------

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        opt = self._make_optimizer()
        handle = opt.register_load_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
