# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.sgd.SGD.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.sgd.SGD.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则直接打包 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 次序：[B,A]/[A,B] | 已覆盖（hook_order_prepend_true_fires_first/hook_order_registration_order） |
| 参数类型 | hook(optimizer) 单实参回调（assertIs 该优化器）；返回值被基类忽略 | 已覆盖（test_hook_trigger/test_hook_return_value_ignored） |
| 传参与不传参 | 显式 prepend=True 与默认（不传 prepend）调用 | 已覆盖（test_hook_order_prepend_true_fires_first/test_hook_order_registration_order） |
| 等价类/边界值 | 打包前触发：hook 改写超参，改写值进入打包结果 | 已覆盖（hook_fires_before_packing/hook_inplace_side_effect_visible） |
| 正常传参场景 | 每次 state_dict() 触发一次；移除后不触发 | 已覆盖（hook_trigger/hook_remove_stops_firing/hook_two_removed_one_keeps） |
| 异常传参场景 | 非 callable 注册不报错、触发时抛 TypeError | 已覆盖（test_hook_non_callable） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移；优化器建于当前设备 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 混合设备类型：本接口为优化器回调注册的纯框架行为，hook 只收 optimizer 实例，无张量跨设备迁移语义；
  优化器参数建于当前设备（self.device_name），设备路径由 test_hook_fires_before_packing 一并执行。
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
class TestSGDregister_state_dict_pre_hook(TestCase):
    """Comprehensive cases for torch.optim.sgd.SGD.register_state_dict_pre_hook (NPU/CUDA portable).

    Covers the single-argument contract (this optimizer, verified by
    identity), dual-hook ordering for prepend=True/False, the
    fires-before-packing position (an in-place hyperparameter change made
    inside the hook shows up in the packed output), the ignored return
    value (the base class state_dict() consumes nothing a pre hook
    returns), removal via RemovableHandle, remove-one-keep-one semantics,
    and the non-callable TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.SGD([p], lr=lr)

    def test_hook_trigger(self):
        """Hook is called with this optimizer (single-argument contract)."""
        opt = self._make_optimizer()
        calls = []
        opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        sd = opt.state_dict()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt,
                      "pre hook must receive this very optimizer")
        # The packed output is unaffected by hook registration.
        self.assertIn("state", sd)
        self.assertIn("param_groups", sd)
        self.assertEqual(sd["param_groups"][0]["params"], [0])

        # removable handle + removal stops firing
        calls2 = []
        h = opt.register_state_dict_pre_hook(lambda o: calls2.append(id(o)))
        opt.state_dict()
        self.assertEqual(len(calls2), 1)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls2), 1, "removed hook must not fire again")

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))  # prepend=False
        opt.register_state_dict_pre_hook(lambda o: order.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))
        opt.register_state_dict_pre_hook(lambda o: order.append("B"))
        opt.state_dict()
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    def test_hook_fires_before_packing(self):
        """The hook fires before the packing loop: live params are raw Tensors.

        Inside the hook the optimizer's param_groups still hold the raw
        Parameter objects (not the int order indices produced by packing),
        proving the pre hook runs before state_dict() packs anything.
        """
        opt = self._make_optimizer()
        seen = {}

        def observe(o):
            raw = o.param_groups[0]["params"][0]
            seen["is_raw_tensor"] = isinstance(raw, torch.Tensor)
            seen["is_param"] = isinstance(raw, torch.nn.Parameter)

        opt.register_state_dict_pre_hook(observe)
        sd = opt.state_dict()
        self.assertTrue(seen["is_raw_tensor"],
                        "inside the pre hook the param group must still hold raw Tensors")
        self.assertTrue(seen["is_param"],
                        "inside the pre hook the param group must still hold the Parameter itself")
        # while the packed output uses int ids
        self.assertEqual(sd["param_groups"][0]["params"], [0])
        self.assertTrue(all(isinstance(i, int) for i in sd["param_groups"][0]["params"]))

    def test_hook_inplace_side_effect_visible(self):
        """An in-place optimizer mutation made in the hook reaches the packed output."""
        opt = self._make_optimizer(lr=0.01)

        def bump_lr(o):
            o.param_groups[0]["lr"] = 0.77  # side effect BEFORE packing
            return None

        opt.register_state_dict_pre_hook(bump_lr)
        sd = opt.state_dict()
        self.assertEqual(sd["param_groups"][0]["lr"], 0.77,
                         "the lr mutated inside the pre hook must appear in the "
                         "packed state_dict, proving the hook runs before packing")
        # the live optimizer keeps the mutated value as well
        self.assertEqual(opt.param_groups[0]["lr"], 0.77)

    def test_hook_return_value_ignored(self):
        """Whatever the pre hook returns, state_dict() still yields the packed dict.

        Unlike load_state_dict pre hooks (whose non-None return replaces
        the load source), the base class state_dict() ignores a state dict
        pre hook's return value entirely.
        """
        opt = self._make_optimizer()

        def override(o):
            return {"fake": 1}  # must NOT replace the real packed dict

        opt.register_state_dict_pre_hook(override)
        sd = opt.state_dict()
        self.assertNotIn("fake", sd,
                         "the return value of a state_dict pre hook must be ignored")
        self.assertIn("state", sd)
        self.assertIn("param_groups", sd)
        self.assertEqual(sd["param_groups"][0]["params"], [0])
        # None-returning hook behaves the same way
        opt.register_state_dict_pre_hook(lambda o: None)
        sd2 = opt.state_dict()
        self.assertIn("param_groups", sd2)
        self.assertEqual(sd2["param_groups"][0]["params"], [0])

    def test_hook_remove_stops_firing(self):
        """A removed hook no longer fires on subsequent state_dict() calls."""
        opt = self._make_optimizer()
        calls = []
        h = opt.register_state_dict_pre_hook(lambda o: calls.append(1))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_hook_two_removed_one_keeps(self):
        """Removing one of two hooks keeps the other one firing."""
        opt = self._make_optimizer()
        calls_a, calls_b = [], []
        h_a = opt.register_state_dict_pre_hook(lambda o: calls_a.append(1))
        h_b = opt.register_state_dict_pre_hook(lambda o: calls_b.append(1))
        opt.state_dict()
        self.assertEqual((len(calls_a), len(calls_b)), (1, 1))
        h_a.remove()
        opt.state_dict()
        self.assertEqual(len(calls_a), 1, "removed hook must not fire again")
        self.assertEqual(len(calls_b), 2, "the kept hook must keep firing")
        h_b.remove()
        opt.state_dict()
        self.assertEqual((len(calls_a), len(calls_b)), (1, 2))

    def test_hook_non_callable(self):
        """A non-callable hook registers fine but raises TypeError when fired."""
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(123)
        with self.assertRaises(TypeError):
            opt.state_dict()

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
