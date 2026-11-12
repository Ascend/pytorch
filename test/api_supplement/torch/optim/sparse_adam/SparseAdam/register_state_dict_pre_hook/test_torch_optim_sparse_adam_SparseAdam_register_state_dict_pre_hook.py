# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.sparse_adam.SparseAdam.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.sparse_adam.SparseAdam.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook: Callable[["Optimizer"], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则无副作用 | 已覆盖（test_hook_trigger/test_hook_no_registration_no_effect） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook [B,A] / 注册顺序 [A,B]） | 已覆盖（test_hook_order_prepend_true/test_hook_order_append） |
| 参数类型 | hook(optimizer) 单实参契约（self 即优化器对象）；返回值被忽略（可返回 dict 也不生效） | 已覆盖（test_hook_trigger/test_hook_return_value_ignored） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true） |
| 等价类/边界值 | 触发时点在 state_dict 打包之前：hook 内改 lr 会体现在打包结果中 | 已覆盖（test_hook_fires_before_packing） |
| 正常传参场景 | 注册后 state_dict() 确实触发；RemovableHandle 可移除且移除后不再触发；二删一留 | 已覆盖（hook_trigger/hook_remove_stops_firing/
remove_one_of_two_keeps_other） |
| 异常传参场景 | 非 callable 注册不抛、触发时 TypeError；None 同理 | 已覆盖（test_non_callable_hook_raises_on_fire） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 混合设备类型：register_state_dict_pre_hook 是纯 Python 回调注册，回调实参为 optimizer 对象，
  无张量跨设备迁移路径；已在 NPU 参数的优化器上验证触发（test_hook_trigger 等），不构造设备对照。
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
class TestSparseAdamregister_state_dict_pre_hook(TestCase):
    """Test cases for torch.optim.sparse_adam.SparseAdam.register_state_dict_pre_hook (NPU/CUDA portable).

    Covers: the (optimizer,) argument contract, firing BEFORE state_dict
    packing, the ignored return value, double-hook ordering for
    prepend=True ([B, A]) and the default append order ([A, B]), handle
    removal, remove-one-keep-other, idempotent double removal, and the
    TypeError raised when a non-callable registration fires.

    Verified contract (upstream optimizer.state_dict): pre hooks receive
    only the optimizer; their return value is discarded; they run before
    param_groups are packed, so mutations made inside them land in the
    packed output.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_opt(self):
        p = torch.nn.Parameter(torch.randn(3, 2, device=self.device), requires_grad=True)
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
        """Hook fires once per state_dict() call with (optimizer,) exactly."""
        calls = []
        opt = self._make_opt()
        opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        sd = opt.state_dict()
        self.assertEqual(len(calls), 1, "hook must fire exactly once per state_dict()")
        self.assertIs(calls[0], opt, "the hook argument must be the optimizer itself")
        self.assertEqual(sorted(sd.keys()), ["param_groups", "state"],
                         "state_dict output stays well-formed after a hook ran")

    def test_hook_single_argument_contract(self):
        """The hook must accept exactly one positional argument; passing more fails."""
        opt = self._make_opt()
        seen = []

        def one_arg(o):
            seen.append(o)

        opt.register_state_dict_pre_hook(one_arg)
        opt.state_dict()
        self.assertEqual(len(seen), 1)
        # A hook with a stricter signature (zero args) cannot be called with
        # the optimizer, which surfaces as a TypeError at fire time.
        opt2 = self._make_opt()
        opt2.register_state_dict_pre_hook(lambda: None)
        with self.assertRaises(TypeError):
            opt2.state_dict()

    def test_hook_no_registration_no_effect(self):
        """Without any registration state_dict() works normally."""
        opt = self._seeded_opt()
        sd = opt.state_dict()
        self.assertEqual(sd["state"][0]["step"], 2)
        self.assertEqual(sd["param_groups"][0]["lr"], 0.05)

    # ------------------------------------------------------------------
    # Fires BEFORE packing
    # ------------------------------------------------------------------

    def test_hook_fires_before_packing(self):
        """Mutations made in the hook land in the packed output (fires before packing)."""
        opt = self._make_opt()

        def bump_lr(o):
            o.param_groups[0]["lr"] = 0.55

        opt.register_state_dict_pre_hook(bump_lr)
        sd = opt.state_dict()
        self.assertEqual(sd["param_groups"][0]["lr"], 0.55,
                         "the pre hook runs before packing, so its mutation is packed")
        self.assertEqual(opt.param_groups[0]["lr"], 0.55)

    # ------------------------------------------------------------------
    # Return value semantics
    # ------------------------------------------------------------------

    def test_hook_return_value_ignored(self):
        """A dict returned by the pre hook is discarded; the packed dict is returned."""
        opt = self._make_opt()
        opt.register_state_dict_pre_hook(lambda o: {"FAKE": 1})
        sd = opt.state_dict()
        self.assertNotIn("FAKE", sd, "the pre hook's return value must be ignored")
        self.assertEqual(sorted(sd.keys()), ["param_groups", "state"])

    # ------------------------------------------------------------------
    # Ordering: prepend=True / default append, two hooks each
    # ------------------------------------------------------------------

    def test_hook_order_prepend_true(self):
        """With prepend=True the later-registered hook fires first ([B, A])."""
        opt = self._make_opt()
        order = []
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))
        opt.register_state_dict_pre_hook(lambda o: order.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the new hook to the front")

    def test_hook_order_append(self):
        """Default registration keeps the registration order ([A, B])."""
        opt = self._make_opt()
        order = []
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))
        opt.register_state_dict_pre_hook(lambda o: order.append("B"))
        opt.state_dict()
        self.assertEqual(order, ["A", "B"],
                         "without prepend the firing order is the registration order")

    # ------------------------------------------------------------------
    # Removal semantics
    # ------------------------------------------------------------------

    def test_hook_remove_stops_firing(self):
        """handle.remove() stops the hook; a double remove is a no-op."""
        opt = self._make_opt()
        calls = []
        h = opt.register_state_dict_pre_hook(lambda o: calls.append(1))
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        h.remove()  # second remove must not raise
        opt.state_dict()
        self.assertEqual(len(calls), 1, "a removed hook must not fire again")

    def test_remove_one_of_two_keeps_other(self):
        """Removing one of two hooks leaves the other one firing."""
        opt = self._make_opt()
        calls = []
        h1 = opt.register_state_dict_pre_hook(lambda o: calls.append("one"))
        opt.register_state_dict_pre_hook(lambda o: calls.append("two"))
        opt.state_dict()
        self.assertEqual(calls, ["one", "two"])
        h1.remove()
        opt.state_dict()
        self.assertEqual(calls, ["one", "two", "two"],
                         "only the removed hook disappears; the survivor keeps firing")

    # ------------------------------------------------------------------
    # Non-callable registrations
    # ------------------------------------------------------------------

    def test_non_callable_hook_raises_on_fire(self):
        """Registering a non-callable succeeds; firing it raises TypeError."""
        opt = self._make_opt()
        opt.register_state_dict_pre_hook(123)
        with self.assertRaises(TypeError):
            opt.state_dict()

    def test_none_hook_raises_on_fire(self):
        """Registering None succeeds; firing it raises TypeError."""
        opt = self._make_opt()
        opt.register_state_dict_pre_hook(None)
        with self.assertRaises(TypeError):
            opt.state_dict()

    # ------------------------------------------------------------------
    # Device-independence sanity
    # ------------------------------------------------------------------

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_opt()
        handle = opt.register_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
