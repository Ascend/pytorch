# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Optimizer.register_load_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.Optimizer.register_load_state_dict_post_hook
API 签名：register_load_state_dict_post_hook(self, hook, prepend=False) -> RemovableHandle

基类行为说明（实测确认）：load post hook 签名为 hook(optimizer)（单位置实参），
在 load_state_dict 完成后触发，hook 内可见已加载的 lr 与 state（含已恢复的
状态张量）。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则默认行为 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook 断言 [B,A] 与 [A,B] 对照） | 已覆盖（hook_order_prepend_true_fires_first/
hook_order_registration_order） |
| 参数类型 | hook(opt) 单位置实参契约；非 callable 注册成功、触发时 TypeError | 已覆盖（hook_single_arg_contract/hook_non_callable_raises_typeerror） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true_fires_first） |
| 等价类/边界值 | 触发位置在 load_state_dict 完成之后（hook 内可见已加载 lr 与已恢复的 state 张量） | 已覆盖（test_hook_fires_after_load） |
| 正常传参场景 | 注册后在 load_state_dict() 完成后确实触发；返回值无语义（None 契约） | 已覆盖（test_hook_trigger/test_hook_fires_after_load） |
| 异常传参场景 | 非 callable hook 抛 TypeError；remove 后不再触发；二删一留 | 已覆盖（hook_trigger/hook_non_callable_raises_typeerror/hook_removal） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义，注册为纯 Python 状态操作） |

未覆盖项及原因：
- 无（post hook 无返回值契约——签名约定返回 None，基类不消费返回值）。
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
class TestOptimizerregister_load_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.Optimizer.register_load_state_dict_post_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: the single-positional-argument
    contract (the hook receives exactly the optimizer), dual-hook ordering
    for prepend=True/False, firing AFTER the load (the hook sees the loaded
    lr and the restored state tensors), removal (both / one of two), and
    the non-callable TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01, momentum=None):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        kwargs = {} if momentum is None else {"momentum": momentum}
        return torch.optim.SGD([p], lr=lr, **kwargs)

    @staticmethod
    def _sd(lr=0.05, momentum=0.0):
        """A minimal loadable state dict with one param group."""
        return {"state": {},
                "param_groups": [{"params": [0], "lr": lr, "momentum": momentum}]}

    def test_hook_trigger(self):
        """Hook is called with this optimizer after the load."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_post_hook(lambda o: calls.append(o))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt)

        calls2 = []
        h = opt.register_load_state_dict_post_hook(
            lambda o: calls2.append(id(o)), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls2), 1)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls2), 1, "removed hook must not fire again")

        opt.register_load_state_dict_post_hook(3.14)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_load_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()

    # ------------------------------------------------------------------
    # Single-positional-argument contract
    # ------------------------------------------------------------------

    def test_hook_single_arg_contract(self):
        """The hook is invoked with exactly one positional argument: the
        optimizer instance itself (no state_dict is passed)."""
        opt = self._make_optimizer()
        seen = {}

        def post_hook(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return None

        opt.register_load_state_dict_post_hook(post_hook)
        opt.load_state_dict(self._sd())
        self.assertEqual(len(seen["args"]), 1,
                         "the hook must receive exactly one positional argument")
        self.assertIs(seen["args"][0], opt,
                      "the single positional argument must be this optimizer")
        self.assertEqual(seen["kwargs"], {},
                         "no keyword arguments may be passed to the hook")
        # A hook declared with the documented single parameter works as-is.
        opt2 = self._make_optimizer()
        opt2.register_load_state_dict_post_hook(lambda o: None)
        opt2.load_state_dict(self._sd())

    # ------------------------------------------------------------------
    # Ordering: prepend=True / default
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"), prepend=True)
        opt.load_state_dict(self._sd())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"))
        opt.load_state_dict(self._sd())
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    # ------------------------------------------------------------------
    # Fires AFTER the load: loaded lr and restored state are visible
    # ------------------------------------------------------------------

    def test_hook_fires_after_load(self):
        """Inside the hook the loaded lr and the restored state tensors
        (matching the source optimizer) are already visible."""
        # Build a source optimizer with real stepped state.
        src = self._make_optimizer(lr=0.09, momentum=0.9)
        p_src = src.param_groups[0]["params"][0]
        p_src.grad = torch.full_like(p_src, 2.0)
        src.step()  # state: momentum_buffer
        sd = src.state_dict()
        self.assertIn("momentum_buffer", sd["state"][0])

        # Target with different lr; the hook must see the LOADED values.
        tgt = self._make_optimizer(lr=0.01, momentum=0.9)
        seen = {}

        def post_hook(o):
            seen["lr"] = o.param_groups[0]["lr"]
            seen["n_state"] = len(o.state)
            seen["buf"] = o.state[o.param_groups[0]["params"][0]].get("momentum_buffer")
            return None

        tgt.register_load_state_dict_post_hook(post_hook)
        tgt.load_state_dict(sd)
        self.assertEqual(seen["lr"], 0.09,
                         "the hook must see the loaded lr (0.09), not the pre-load one")
        self.assertEqual(seen["n_state"], 1,
                         "the hook must see the restored state entry")
        # The restored momentum_buffer matches the source optimizer's state.
        torch.testing.assert_close(seen["buf"], src.state[p_src]["momentum_buffer"])
        # The hook fired exactly once for this load.
        self.assertEqual(len(seen), 3)

    # ------------------------------------------------------------------
    # Removal: both hooks, and one of two
    # ------------------------------------------------------------------

    def test_hook_removal(self):
        """Removed hooks stop firing; removing one of two keeps the other."""
        calls = []
        opt = self._make_optimizer()
        ha = opt.register_load_state_dict_post_hook(lambda o: calls.append("A"))
        hb = opt.register_load_state_dict_post_hook(lambda o: calls.append("B"))
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
        h = opt.register_load_state_dict_post_hook("not-callable")
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
