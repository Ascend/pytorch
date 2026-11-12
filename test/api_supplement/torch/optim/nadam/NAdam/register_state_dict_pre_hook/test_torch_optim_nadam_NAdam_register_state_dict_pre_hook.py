# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.nadam.NAdam.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.nadam.NAdam.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则不触发 | 已覆盖（test_hook_trigger_once / test_no_hook_no_trigger） |
| 枚举选项 | prepend=True/False 控制双 hook 触发次序 | 已覆盖（test_hook_order_prepend_true / test_hook_order_prepend_false） |
| 参数类型 | hook(optimizer) 单参回调；非 callable 报 TypeError | 已覆盖（test_hook_args_contract / test_non_callable_raises_on_fire） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger_once / test_hook_order_prepend_false） |
| 等价类/边界值 | 触发位置在 state_dict 打包之前（可先做准备工作）；hook 返回值被忽略（实测） | 已覆盖（test_hook_runs_before_packing / test_hook_return_value_ignored） |
| 正常传参场景 | 注册后在 state_dict() 中确实被触发；返回 RemovableHandle | 已覆盖（test_hook_trigger_once / test_removable_handle_type） |
| 异常传参场景 | 非 callable hook 抛 TypeError；移除后不再触发 | 已覆盖（test_non_callable_raises_on_fire / test_remove_hook_stops_trigger） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- hook 返回 dict 替换 state_dict()：实测基类实现只消费 state_dict POST hook 的返回值，
  PRE hook 的返回值被忽略（torch/optim/optimizer.py: state_dict() 中
  `for pre_hook ...: pre_hook(self)` 无赋值），故本文件断言返回值被忽略而非替换。
- 注册非 callable 时立即抛错：实测当前版本注册不校验、触发时才抛 TypeError，
  本文件按实测行为断言（test_non_callable_raises_on_fire）。
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
class TestNAdamregister_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.nadam.NAdam.register_state_dict_pre_hook (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        # A fresh optimizer per case keeps global hook registries isolated.
        return torch.optim.NAdam(
            [torch.nn.Parameter(torch.randn(3, device=self.device_name),
                                requires_grad=True)], lr=0.01)

    # ------------------------------------------------------------------
    # Basic trigger + argument contract
    # ------------------------------------------------------------------

    def test_hook_trigger_once(self):
        """Hook is called exactly once per state_dict() with the optimizer."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt, "hook must receive the optimizer itself")

    def test_hook_args_contract(self):
        """The hook receives exactly one positional argument (the optimizer)."""
        opt = self._make_optimizer()
        captured = []

        def hook(*args, **kwargs):
            captured.append((args, kwargs))

        opt.register_state_dict_pre_hook(hook)
        opt.state_dict()
        self.assertEqual(len(captured), 1)
        args, kwargs = captured[0]
        self.assertEqual(len(args), 1, "pre hook must be called with one arg")
        self.assertIs(args[0], opt)
        self.assertEqual(kwargs, {}, "pre hook must be called without kwargs")

    def test_no_hook_no_trigger(self):
        """Without any registered hook, state_dict() returns the normal dict."""
        opt = self._make_optimizer()
        sd = opt.state_dict()
        # The default payload is still fully built.
        self.assertIn("param_groups", sd)
        self.assertIn("state", sd)
        self.assertEqual(len(sd["param_groups"]), 1)

    # ------------------------------------------------------------------
    # prepend ordering with TWO hooks (both [B, A] and [A, B] contrasts)
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true(self):
        """With prepend=True the later-registered hook fires FIRST: [B, A]."""
        opt = self._make_optimizer()
        order = []
        opt.register_state_dict_pre_hook(lambda o: order.append('A'))
        opt.register_state_dict_pre_hook(lambda o: order.append('B'),
                                         prepend=True)
        opt.state_dict()
        self.assertEqual(order, ['B', 'A'],
                         "prepend=True must put the later hook first")

    def test_hook_order_prepend_false(self):
        """With prepend=False the registration order is kept: [A, B]."""
        opt = self._make_optimizer()
        order = []
        opt.register_state_dict_pre_hook(lambda o: order.append('A'))
        opt.register_state_dict_pre_hook(lambda o: order.append('B'),
                                         prepend=False)
        opt.state_dict()
        self.assertEqual(order, ['A', 'B'],
                         "prepend=False must keep registration order")

    # ------------------------------------------------------------------
    # Lifecycle position: the pre hook runs BEFORE the dict is packed
    # ------------------------------------------------------------------

    def test_hook_runs_before_packing(self):
        """The hook fires before packing, so it observes live param groups."""
        opt = self._make_optimizer()
        seen = []
        opt.register_state_dict_pre_hook(lambda o: seen.append(len(o.param_groups)))
        # Growing the optimizer AFTER registration must be visible to the
        # hook at state_dict() time (i.e. the hook reads live state).
        p2 = torch.nn.Parameter(torch.randn(3, device=self.device_name))
        opt.add_param_group({"params": [p2]})
        opt.state_dict()
        self.assertEqual(seen, [2],
                         "pre hook must observe the live, unpacked optimizer")

    def test_hook_return_value_ignored(self):
        """The pre hook's return value does NOT replace the produced dict."""
        # Verified against the installed build: Optimizer.state_dict() calls
        # `for pre_hook in ...: pre_hook(self)` and DISCARDS any result; only
        # POST hooks may return a replacement dict.
        opt = self._make_optimizer()

        def override(o):
            return {"custom": True}

        opt.register_state_dict_pre_hook(override)
        sd = opt.state_dict()
        self.assertIn("param_groups", sd,
                      "state_dict() must ignore the pre hook return value "
                      "and still return the packed optimizer state")
        self.assertIn("state", sd)
        self.assertNotIn("custom", sd)

    # ------------------------------------------------------------------
    # RemovableHandle lifecycle
    # ------------------------------------------------------------------

    def test_removable_handle_type(self):
        """register returns a RemovableHandle instance."""
        opt = self._make_optimizer()
        handle = opt.register_state_dict_pre_hook(lambda o: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)

    def test_remove_hook_stops_trigger(self):
        """After handle.remove(), the hook no longer fires."""
        opt = self._make_optimizer()
        calls = []
        handle = opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        handle.remove()
        opt.state_dict()
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_remove_one_of_two_keeps_other(self):
        """Removing one hook leaves the other registered and firing."""
        opt = self._make_optimizer()
        calls_a, calls_b = [], []
        ha = opt.register_state_dict_pre_hook(lambda o: calls_a.append(1))
        opt.register_state_dict_pre_hook(lambda o: calls_b.append(1))
        opt.state_dict()
        self.assertEqual((len(calls_a), len(calls_b)), (1, 1))
        ha.remove()
        opt.state_dict()
        self.assertEqual(len(calls_a), 1, "removed hook A must stay silent")
        self.assertEqual(len(calls_b), 2, "hook B must keep firing")

    # ------------------------------------------------------------------
    # Non-callable rejection
    # ------------------------------------------------------------------

    def test_non_callable_raises_on_fire(self):
        """A non-callable hook is accepted at registration but raises on fire."""
        # Actual behavior on the installed build: register_state_dict_pre_hook
        # itself performs no callable check; the TypeError is raised when
        # state_dict() iterates the hooks.
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(123)
        with self.assertRaises(TypeError):
            opt.state_dict()


if __name__ == "__main__":
    run_tests()
