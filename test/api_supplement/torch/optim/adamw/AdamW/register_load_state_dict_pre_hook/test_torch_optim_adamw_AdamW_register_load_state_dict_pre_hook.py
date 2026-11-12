# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adamw.AdamW.register_load_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.adamw.AdamW.register_load_state_dict_pre_hook
API 签名：register_load_state_dict_pre_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空优化器与 step 产生非空 state 后均触发 | 已覆盖（test_pre_hook_two_arg_contract / test_pre_hook_return_dict_replaces_source） |
| 枚举选项 | prepend=True/False 双次序 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 参数类型 | hook 双实参回调；非 callable 触发 TypeError | 已覆盖（pre_hook_two_arg_contract/non_callable_hook_raises_on_fire） |
| 传参与不传参 | 显式 prepend=True 与省略 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 等价类/边界值 | 触发点在加载处理前：in-place 改写 incoming dict 生效；返回 dict 完全替换加载源；返回 None 用原始 | 已覆盖（pre_hook_inplace_modify_incoming_dict/
pre_hook_return_dict_replaces_source/pre_hook_return_none_uses_original） |
| 正常传参场景 | 注册 → load 触发一次；返回 RemovableHandle | 已覆盖（test_pre_hook_two_arg_contract / test_remove_stops_firing） |
| 异常传参场景 | 非 callable hook 注册不抛错、触发时抛 TypeError；对照组证明坏 dict 无 hook 时 ValueError | 已覆盖（non_callable_hook_raises_on_fire/
bad_incoming_dict_without_hook_raises） |
| 移除语义 | remove 后不再触发；二删一留 | 已覆盖（test_remove_stops_firing / test_remove_one_of_two_keeps_other） |
| 混合设备类型 | 纯框架回调注册，参数张量位于当前设备（NPU/CUDA） | 已覆盖（全部用例在 self.device 上构造参数） |

未覆盖项及原因：
- 本 hook 返回值语义（替换加载源）与 in-place 改写为主要契约，已在等价类/边界值行全覆盖；
  数值执行语义由 AdamW/step 用例覆盖，本文件不重复。
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
class TestAdamWregister_load_state_dict_pre_hook(TestCase):
    """Comprehensive cases for torch.optim.adamw.AdamW.register_load_state_dict_pre_hook (NPU/CUDA portable).

    AdamW inherits the hook machinery from Optimizer, but this file still covers
    the full contract independently: the two-positional-arg contract
    (optimizer, incoming state dict), prepend ordering with two hooks
    ([B, A] vs [A, B]), the in-place rewrite path, the return-dict path
    (the return value fully replaces the load source), the return-None path,
    removal semantics and the non-callable TypeError-on-fire path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.1):
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        return torch.optim.AdamW([p], lr=lr), p

    def _make_stepped_optimizer(self, lr=0.1):
        """Step once so the optimizer carries real AdamW state tensors."""
        opt, p = self._make_optimizer(lr)
        p.grad = torch.ones_like(p)
        opt.step()
        return opt, p

    # ------------------------------------------------------------------
    # Argument contract
    # ------------------------------------------------------------------

    def test_pre_hook_two_arg_contract(self):
        """The hook receives exactly (optimizer, incoming_state_dict)."""
        opt, _ = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_pre_hook(lambda *a, **k: calls.append((a, k)))
        incoming = opt.state_dict()
        opt.load_state_dict(incoming)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(len(args), 2, "the hook must receive two positional args")
        self.assertIs(args[0], opt, "first positional arg must be the optimizer itself")
        self.assertIsInstance(args[1], dict, "second positional arg must be the state dict")
        self.assertEqual(kwargs, {}, "no keyword arguments are passed to the hook")
        # The incoming dict is a shallow copy: same content, different outer dict.
        self.assertIsNot(args[1], incoming)
        self.assertEqual(args[1], incoming)

    # ------------------------------------------------------------------
    # Load-source manipulation: in-place rewrite, return-replaces, return-None
    # ------------------------------------------------------------------

    def test_pre_hook_inplace_modify_incoming_dict(self):
        """In-place edits to the incoming dict take effect without returning it."""
        p1 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(2, device=self.device), requires_grad=True)
        opt = torch.optim.AdamW([{"params": [p1], "lr": 0.1},
                                {"params": [p2], "lr": 0.2}])

        def rewrite(o, sd):
            for g in sd["param_groups"]:
                g["lr"] = 0.9
            return None  # no return: only the in-place edits apply

        opt.register_load_state_dict_pre_hook(rewrite)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual([g["lr"] for g in opt.param_groups], [0.9, 0.9],
                         "in-place lr rewrite must affect both loaded groups")

    def test_pre_hook_return_dict_replaces_source(self):
        """A returned dict fully replaces the incoming dict as the load source."""
        opt, p = self._make_stepped_optimizer(lr=0.1)
        self.assertTrue(len(opt.state) > 0, "precondition: non-empty state")
        replacement = {"state": {}, "param_groups": [{"params": [0], "lr": 0.5}]}
        opt.register_load_state_dict_pre_hook(lambda o, sd: replacement)
        # The incoming dict is unusable (zero groups); only the returned dict
        # can load. If the return value were ignored this raises ValueError.
        opt.load_state_dict({"state": {}, "param_groups": []})
        self.assertEqual(opt.param_groups[0]["lr"], 0.5,
                         "the returned dict must be the loaded source")
        # The replacement's empty state was loaded too (full replacement).
        self.assertEqual(len(opt.state), 0,
                         "the replacement's empty state must be loaded")

    def test_bad_incoming_dict_without_hook_raises(self):
        """Control: without a hook the same unusable dict raises ValueError."""
        opt, _ = self._make_optimizer(lr=0.1)
        with self.assertRaisesRegex(ValueError,
                                    "different number of parameter groups"):
            opt.load_state_dict({"state": {}, "param_groups": []})

    def test_pre_hook_return_none_uses_original(self):
        """Returning None keeps the incoming dict as the load source."""
        opt, _ = self._make_optimizer(lr=0.1)
        opt.register_load_state_dict_pre_hook(lambda o, sd: None)
        sd = opt.state_dict()
        opt.param_groups[0]["lr"] = 0.3  # diverged; the load must restore 0.1
        opt.load_state_dict(sd)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1,
                         "returning None must load the original incoming dict")

    # ------------------------------------------------------------------
    # Ordering: prepend=True vs default registration order (two hooks)
    # ------------------------------------------------------------------

    def test_order_prepend_true_fires_new_hook_first(self):
        """prepend=True puts the newly registered hook before existing ones: [B, A]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must fire B before the previously registered A")

    def test_order_default_follows_registration_order(self):
        """Default (prepend omitted/False) fires hooks in registration order: [A, B]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("A"))
        opt.register_load_state_dict_pre_hook(lambda o, sd: order.append("B"), prepend=False)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["A", "B"],
                         "without prepend the registration order must be kept")

    # ------------------------------------------------------------------
    # Removal semantics
    # ------------------------------------------------------------------

    def test_remove_stops_firing(self):
        """A removed handle never fires again on subsequent loads."""
        opt, _ = self._make_optimizer()
        calls = []
        h = opt.register_load_state_dict_pre_hook(lambda o, sd: calls.append(o))
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        sd = opt.state_dict()
        opt.load_state_dict(sd)
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.load_state_dict(sd)
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_remove_one_of_two_keeps_other(self):
        """Removing one of two hooks leaves the other fully functional."""
        opt, _ = self._make_optimizer()
        c1, c2 = [], []
        h1 = opt.register_load_state_dict_pre_hook(lambda o, sd: c1.append(1))
        opt.register_load_state_dict_pre_hook(lambda o, sd: c2.append(1))
        sd = opt.state_dict()
        opt.load_state_dict(sd)
        self.assertEqual((len(c1), len(c2)), (1, 1))
        h1.remove()
        opt.load_state_dict(sd)
        self.assertEqual((len(c1), len(c2)), (1, 2),
                         "removed hook stays silent while the kept hook still fires")

    # ------------------------------------------------------------------
    # Non-callable hooks
    # ------------------------------------------------------------------

    def test_non_callable_hook_raises_on_fire(self):
        """Non-callable hooks register without error but raise TypeError when fired."""
        opt, _ = self._make_optimizer()
        for bad in (None, 3.14, "not-callable"):
            opt.register_load_state_dict_pre_hook(bad)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    # ------------------------------------------------------------------
    # Handle type
    # ------------------------------------------------------------------

    def test_handle_type(self):
        """Registration returns a RemovableHandle bound to this optimizer."""
        opt, _ = self._make_optimizer()
        handle = opt.register_load_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
