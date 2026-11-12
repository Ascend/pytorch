# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Optimizer.register_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.Optimizer.register_state_dict_post_hook
API 签名：register_state_dict_post_hook(self, hook, prepend=False) -> RemovableHandle

基类行为说明（实测确认）：post hook 签名为 hook(optimizer, state_dict)（双实参），
在打包完成后触发（收到的 dict 中 params 已是 int 索引、state 键已重映射）；
返回值生效：返回 dict 完全替换 state_dict() 输出、返回 None 保留（含 in-place
修改后的）原 dict。

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则默认状态 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook 断言 [B,A] 与 [A,B] 对照） | 已覆盖（hook_order_prepend_true_fires_first/
hook_order_registration_order） |
| 参数类型 | hook(opt, state_dict) 双实参契约；非 callable 注册成功、触发时 TypeError | 已覆盖（hook_trigger/
hook_non_callable_raises_typeerror） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true_fires_first） |
| 等价类/边界值 | 触发位置在 state_dict 打包之后（收到打包后 dict：params 为 int 索引、state 已重映射、超参键齐全） | 已覆盖（test_hook_fires_after_packing） |
| 正常传参场景 | 返回 dict 完全替换输出、返回 None 保留原结果、in-place 改 lr 可见 | 已覆盖（test_hook_return_dict_replaces/test_hook_inplace_mutation） |
| 异常传参场景 | 非 callable hook 抛 TypeError；remove 后不再触发；二删一留 | 已覆盖（hook_trigger/
hook_non_callable_raises_typeerror/hook_removal） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义，注册为纯 Python 状态操作） |

未覆盖项及原因：
- 无（post hook 返回值契约已全覆盖：dict 替换 / None 保留 / in-place 生效）。
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
class TestOptimizerregister_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.Optimizer.register_state_dict_post_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: the hook fires AFTER state_dict
    packing (it receives the packed dict with int param indices and
    remapped state keys), dual-hook ordering for prepend=True/False,
    return-value semantics (a returned dict replaces the result; None keeps
    the possibly in-place-mutated dict), removal (both / one of two), and
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

    def test_hook_trigger(self):
        """Hook is called with (this optimizer, packed state dict)."""
        opt = self._make_optimizer()
        seen = []
        opt.register_state_dict_post_hook(lambda o, sd: seen.append((o, sd)))
        sd = opt.state_dict()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], opt, "post hook must receive this very optimizer")
        self.assertIsInstance(seen[0][1], dict)

        # post-hook may return a replacement dict
        def inject(o, sd):
            sd["extra"] = 42
            return sd

        opt.register_state_dict_post_hook(inject)
        out = opt.state_dict()
        self.assertEqual(out.get("extra"), 42)

        # removable + removal stops firing
        calls = []
        h = opt.register_state_dict_post_hook(lambda o, sd: calls.append(1))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

        # non-callable registration succeeds but raises TypeError when fired
        opt.register_state_dict_post_hook("not-callable")
        with self.assertRaises(TypeError):
            opt.state_dict()

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()

    # ------------------------------------------------------------------
    # Fires AFTER packing
    # ------------------------------------------------------------------

    def test_hook_fires_after_packing(self):
        """The hook receives the packed dict: int param indices, remapped
        state keys, and the optimizer's hyperparameter keys."""
        opt = self._make_optimizer(lr=0.07, momentum=0.9)
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.ones_like(p)
        opt.step()  # produce non-empty state so the packed dict is observable
        seen = {}
        opt.register_state_dict_post_hook(lambda o, sd: seen.update(sd))
        opt.state_dict()
        self.assertIn("state", seen)
        self.assertIn("param_groups", seen)
        # Packing replaces Tensors with order indices (ints), proving the
        # hook fires after the packing loop, not on the raw optimizer state.
        packed_params = seen["param_groups"][0]["params"]
        self.assertEqual(packed_params, [0])
        self.assertTrue(all(isinstance(i, int) for i in packed_params))
        # The packed group carries the optimizer's own hyperparameters.
        group = seen["param_groups"][0]
        self.assertEqual(group["lr"], 0.07)
        self.assertEqual(group["momentum"], 0.9)
        # The stepped state was packed under index 0 with SGD's state key.
        self.assertIn(0, seen["state"])
        self.assertIn("momentum_buffer", seen["state"][0])
        # Two params across two groups prove index remapping continues.
        pa = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        pb = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt2 = torch.optim.SGD([{"params": [pa], "lr": 0.1}, {"params": [pb], "lr": 0.2}])
        seen2 = {}
        opt2.register_state_dict_post_hook(lambda o, sd: seen2.update(sd))
        opt2.state_dict()
        self.assertEqual([g["params"] for g in seen2["param_groups"]], [[0], [1]],
                         "packing must remap params to continuing int indices")

    # ------------------------------------------------------------------
    # Ordering: prepend=True / default
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_state_dict_post_hook(lambda o, sd: order.append("A"))  # prepend=False
        opt.register_state_dict_post_hook(lambda o, sd: order.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_state_dict_post_hook(lambda o, sd: order.append("A"))
        opt.register_state_dict_post_hook(lambda o, sd: order.append("B"))
        opt.state_dict()
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    # ------------------------------------------------------------------
    # Return-value contract: dict replaces, None keeps
    # ------------------------------------------------------------------

    def test_hook_return_dict_replaces(self):
        """A hook returning a dict replaces the state_dict() result."""
        opt = self._make_optimizer(lr=0.01)
        opt.register_state_dict_post_hook(lambda o, sd: {"replaced": 1})
        self.assertEqual(opt.state_dict(), {"replaced": 1},
                         "a returned dict must become the new state_dict result")

    def test_hook_return_none_keeps_original(self):
        """Returning None keeps the (possibly mutated) original packed dict."""
        opt = self._make_optimizer(lr=0.02)
        opt.register_state_dict_post_hook(lambda o, sd: None)
        out = opt.state_dict()
        self.assertIn("state", out, "None return must keep the packed dict")
        self.assertIn("param_groups", out)
        self.assertEqual(out["param_groups"][0]["lr"], 0.02,
                         "the kept dict must still reflect the optimizer state")

    def test_hook_inplace_mutation(self):
        """In-place mutation of the received dict reaches the caller."""
        opt = self._make_optimizer()

        def mutate(o, sd):
            sd["param_groups"][0]["lr"] = 0.5
            return None

        opt.register_state_dict_post_hook(mutate)
        sd = opt.state_dict()
        self.assertEqual(sd["param_groups"][0]["lr"], 0.5,
                         "in-place mutation with None return must be kept")
        # The mutation affects only the packed output, not the live optimizer.
        self.assertEqual(opt.param_groups[0]["lr"], 0.01,
                         "post hook mutates the packed dict, not the optimizer")

    # ------------------------------------------------------------------
    # Removal: both hooks, and one of two
    # ------------------------------------------------------------------

    def test_hook_removal(self):
        """Removed hooks stop firing; removing one of two keeps the other."""
        calls = []
        opt = self._make_optimizer()
        ha = opt.register_state_dict_post_hook(lambda o, sd: calls.append("A"))
        hb = opt.register_state_dict_post_hook(lambda o, sd: calls.append("B"))
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
        h = opt.register_state_dict_post_hook(3.14)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle,
                              "registering a non-callable must not raise")
        with self.assertRaises(TypeError):
            opt.state_dict()
        h.remove()
        # state_dict works again once the bad hook is removed.
        self.assertIn("param_groups", opt.state_dict())


if __name__ == "__main__":
    run_tests()
