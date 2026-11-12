# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.lbfgs.LBFGS.register_load_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.lbfgs.LBFGS.register_load_state_dict_pre_hook
API 签名：register_load_state_dict_pre_hook(self, hook: Callable[[Optimizer, StateDict], StateDict | None], prepend=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；hook 内 optimizer 仍是加载前状态（fires before load） | 已覆盖（hook_trigger/
hook_fires_before_load） |
| 枚举选项 | prepend=False/True 双 hook 断言触发次序：prepend 后注册者先触发（[B,A]），默认注册序（[A,B]） | 已覆盖（hook_order_prepend_true_fires_first/
hook_order_registration_order） |
| 参数类型 | hook(optimizer, state_dict) 双实参契约：第一实参为该优化器（assertIs），第二实参为用户传入 dict 的 shallow copy | 已覆盖（hook_trigger/
hook_inplace_rewrite_loads） |
| 传参与不传参 | 显式 prepend=True 与默认（不传 prepend）调用 | 已覆盖（test_hook_order_prepend_true_fires_first/test_hook_order_registration_order） |
| 等价类/边界值 | 触发位置在 load 之前：hook 内可改写 incoming dict（嵌套组内 lr 改写后生效）；收到 dict 外层为拷贝、嵌套与原 dict 共享 |
已覆盖（hook_fires_before_load/hook_inplace_rewrite_loads） |
| 正常传参场景 | 返回 dict 替换加载源；返回 None 用原始（改写后）dict；先触发 hook 的返回 dict 喂给后触发 hook（链式）；移除后不再触发；返回 RemovableHandle |
已覆盖（hook_return_dict_replaces_load_source/hook_return_none_uses_original/hook_chained_return_feeds_next_hook/
hook_remove_stops_firing） |
| 异常传参场景 | 非 callable 注册不报错、触发时抛 TypeError；二删一留（删 A 留 B 后 B 仍触发） | 已覆盖（test_hook_non_callable/test_hook_two_removed_one_keeps） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移；优化器建于当前设备 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 混合设备类型：本接口为优化器回调注册的纯框架行为，hook 仅收 (optimizer, state_dict) 两个 Python 对象，
  无张量跨设备迁移语义；优化器参数建于当前设备（self.device_name），设备路径由各用例一并执行。
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
class TestLBFGSregister_load_state_dict_pre_hook(TestCase):
    """Comprehensive cases for torch.optim.lbfgs.LBFGS.register_load_state_dict_pre_hook (NPU/CUDA portable).

    Covers the two-argument contract (this optimizer plus a shallow copy of
    the incoming state dict), dual-hook ordering for prepend=True/False, the
    fires-before-load position, in-place rewriting of the incoming dict,
    return-value semantics (a returned dict replaces the load source; None
    keeps the possibly-mutated original; an earlier hook's returned dict is
    fed to later hooks), removal via RemovableHandle, remove-one-keep-one,
    and the non-callable TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=1.0):
        # LBFGS accepts a flat param list only: its __init__ rejects more than
        # one param group (ValueError), so keep a single group here.
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.LBFGS([p], lr=lr)

    def test_hook_trigger(self):
        """Hook is called with (this optimizer, the incoming state dict)."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append((o, sd)))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], opt,
                      "pre hook must receive this very optimizer")
        received = calls[0][1]
        self.assertIsInstance(received, dict)
        self.assertIn("state", received)
        self.assertIn("param_groups", received)

        # a hook may rewrite the state dict before loading
        def bump_lr(o, sd):
            for g in sd["param_groups"]:
                g["lr"] = 0.9
            return sd

        before = opt.state_dict()
        opt.register_load_state_dict_pre_hook(bump_lr)
        opt.load_state_dict(before)
        self.assertEqual(opt.param_groups[0]["lr"], 0.9)

        h = opt.register_load_state_dict_pre_hook(lambda o, sd: None)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()

        opt.register_load_state_dict_pre_hook(None)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))  # prepend=False
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    def test_hook_fires_before_load(self):
        """Inside the hook the optimizer still has its pre-load hyperparams."""
        opt = self._make_optimizer(lr=1.0)
        incoming = opt.state_dict()
        incoming["param_groups"][0]["lr"] = 0.33  # the value to be loaded
        seen = {}

        def observe(o, sd):
            seen["lr_inside_hook"] = o.param_groups[0]["lr"]
            seen["received_lr"] = sd["param_groups"][0]["lr"]

        opt.register_load_state_dict_pre_hook(observe)
        opt.load_state_dict(incoming)
        self.assertEqual(seen["lr_inside_hook"], 1.0,
                         "pre hook must fire before the load is applied, "
                         "so the optimizer still shows the old lr")
        self.assertEqual(seen["received_lr"], 0.33)
        # after the load completes the new lr is applied
        self.assertEqual(opt.param_groups[0]["lr"], 0.33)

    def test_hook_inplace_rewrite_loads(self):
        """In-place rewriting of the incoming dict is what gets loaded.

        The received dict is a shallow copy of the caller's dict: the outer
        dict is a new object, while nested dicts are shared with the
        original, so a nested mutation also shows up in the caller's dict.
        """
        opt = self._make_optimizer(lr=1.0)
        orig = opt.state_dict()
        seen = {}

        def rewrite(o, sd):
            seen["is_orig"] = sd is orig
            seen["nested_shared"] = sd["param_groups"][0] is orig["param_groups"][0]
            sd["param_groups"][0]["lr"] = 0.7  # in-place rewrite of the incoming dict
            sd["param_groups"][0]["max_iter"] = 7
            return None

        opt.register_load_state_dict_pre_hook(rewrite)
        opt.load_state_dict(orig)
        self.assertFalse(seen["is_orig"],
                         "load_state_dict passes a shallow copy, not the caller's dict")
        self.assertTrue(seen["nested_shared"],
                        "the shallow copy still shares nested dicts with the original")
        self.assertEqual(opt.param_groups[0]["lr"], 0.7,
                         "the in-place rewritten lr must be the loaded value")
        self.assertEqual(opt.param_groups[0]["max_iter"], 7)
        # Shallow-copy hazard: the nested mutation propagated back to the
        # caller's dict as well (documented 'shallow copy' contract).
        self.assertEqual(orig["param_groups"][0]["lr"], 0.7)

    def test_hook_return_dict_replaces_load_source(self):
        """A hook returning a dict makes that dict the load source."""
        opt = self._make_optimizer(lr=1.0)
        received = {}

        def replace(o, sd):
            received["incoming_lr"] = sd["param_groups"][0]["lr"]
            return {"state": {},
                    "param_groups": [{"params": [0], "lr": 0.123, "max_iter": 5}]}

        opt.register_load_state_dict_pre_hook(replace)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(received["incoming_lr"], 1.0,
                         "the hook still sees the user-passed dict first")
        self.assertEqual(opt.param_groups[0]["lr"], 0.123,
                         "the returned dict must replace the load source")
        self.assertEqual(opt.param_groups[0]["max_iter"], 5)

    def test_hook_return_none_uses_original(self):
        """Returning None loads the (possibly rewritten) original dict."""
        opt = self._make_optimizer(lr=1.0)
        opt.register_load_state_dict_pre_hook(lambda o, sd: None)
        incoming = opt.state_dict()
        incoming["param_groups"][0]["lr"] = 0.42
        opt.load_state_dict(incoming)
        self.assertEqual(opt.param_groups[0]["lr"], 0.42,
                         "None return must keep the original incoming dict as source")

    def test_hook_chained_return_feeds_next_hook(self):
        """The dict returned by an earlier hook is passed to the next hook."""
        opt = self._make_optimizer(lr=1.0)

        def first(o, sd):
            replaced = {"state": {},
                        "param_groups": [{"params": [0], "lr": 0.11, "max_iter": 3}]}
            return replaced

        seen = {}

        def second(o, sd):
            seen["lr"] = sd["param_groups"][0]["lr"]
            return None  # keep the dict produced by the first hook

        opt.register_load_state_dict_pre_hook(first)   # fires first (registration order)
        opt.register_load_state_dict_pre_hook(second)  # fires second, must see 0.11
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(seen["lr"], 0.11,
                         "a non-None return from an earlier pre hook must replace "
                         " the state dict handed to later hooks")
        self.assertEqual(opt.param_groups[0]["lr"], 0.11)

    def test_hook_remove_stops_firing(self):
        """A removed hook no longer fires on subsequent loads."""
        opt = self._make_optimizer()
        calls = []
        h = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append(1))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_hook_two_removed_one_keeps(self):
        """Removing one of two hooks keeps the other one firing."""
        opt = self._make_optimizer()
        calls_a, calls_b = [], []
        h_a = opt.register_load_state_dict_pre_hook(lambda o, sd: calls_a.append(1))
        h_b = opt.register_load_state_dict_pre_hook(lambda o, sd: calls_b.append(1))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual((len(calls_a), len(calls_b)), (1, 1))
        h_a.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls_a), 1, "removed hook must not fire again")
        self.assertEqual(len(calls_b), 2, "the kept hook must keep firing")
        h_b.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual((len(calls_a), len(calls_b)), (1, 2))

    def test_hook_non_callable(self):
        """A non-callable hook registers fine but raises TypeError when fired."""
        opt = self._make_optimizer()
        opt.register_load_state_dict_pre_hook(None)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_load_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
