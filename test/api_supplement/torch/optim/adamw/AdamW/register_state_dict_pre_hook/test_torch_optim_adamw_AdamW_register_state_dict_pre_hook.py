# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adamw.AdamW.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.adamw.AdamW.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空优化器与注册后新增参数组（live 2 组）均触发 | 已覆盖（test_pre_hook_single_arg_contract / test_pre_hook_fires_before_packing_sees_live_groups） |
| 枚举选项 | prepend=True/False 双次序 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 参数类型 | hook 单实参回调；非 callable 触发 TypeError | 已覆盖（pre_hook_single_arg_contract/non_callable_hook_raises_on_fire） |
| 传参与不传参 | 显式 prepend=True 与省略 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 等价类/边界值 | 打包前触发：hook 读 live 优化器（含后加组）；返回值被忽略；in-place 副作用可见 | 已覆盖（pre_hook_fires_before_packing_sees_live_groups/
pre_hook_return_value_ignored/pre_hook_inplace_side_effect_visible） |
| 正常传参场景 | 注册 → state_dict() 触发一次；返回 RemovableHandle | 已覆盖（test_pre_hook_single_arg_contract / test_remove_stops_firing） |
| 异常传参场景 | 非 callable hook 注册不抛错、触发时抛 TypeError | 已覆盖（test_non_callable_hook_raises_on_fire） |
| 移除语义 | remove 后不再触发；二删一留 | 已覆盖（test_remove_stops_firing / test_remove_one_of_two_keeps_other） |
| 混合设备类型 | 纯框架回调注册，参数张量位于当前设备（NPU/CUDA） | 已覆盖（全部用例在 self.device 上构造参数） |

未覆盖项及原因：
- hook 返回值分支：本 hook 契约为 hook(optimizer) -> None，Optimizer.state_dict 只以
  pre_hook(self) 调用且不消费返回值；已在 test_pre_hook_return_value_ignored 中实证
  返回 dict 被忽略、state_dict() 仍返回打包 dict。
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
class TestAdamWregister_state_dict_pre_hook(TestCase):
    """Comprehensive cases for torch.optim.adamw.AdamW.register_state_dict_pre_hook (NPU/CUDA portable).

    AdamW inherits the hook machinery from Optimizer, but this file still covers
    the full contract independently: the single-positional-arg contract, the
    fires-before-packing timing (the hook reads the LIVE optimizer, so groups
    added after registration are seen), the ignored-return-value contract,
    the in-place side-effect path, prepend ordering with two hooks
    ([B, A] vs [A, B]), removal semantics and the non-callable
    TypeError-on-fire path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.1):
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        return torch.optim.AdamW([p], lr=lr), p

    # ------------------------------------------------------------------
    # Argument contract and firing point
    # ------------------------------------------------------------------

    def test_pre_hook_single_arg_contract(self):
        """The hook receives exactly one positional argument: the optimizer itself."""
        opt, _ = self._make_optimizer()
        calls = []
        opt.register_state_dict_pre_hook(lambda *a, **k: calls.append((a, k)))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(len(args), 1, "the hook must receive a single positional arg")
        self.assertIs(args[0], opt, "the positional arg must be the optimizer itself")
        self.assertEqual(kwargs, {}, "no keyword arguments are passed to the hook")

    def test_pre_hook_fires_before_packing_sees_live_groups(self):
        """The hook reads the live optimizer: groups added AFTER registration are seen."""
        p1 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        p2 = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        opt = torch.optim.AdamW([p1], lr=0.1)
        seen = []
        opt.register_state_dict_pre_hook(
            lambda o: seen.append((len(o.param_groups),
                                   [g["lr"] for g in o.param_groups])))
        # A group added after hook registration must be visible to the hook:
        # the hook fires BEFORE packing, on the live 2-group optimizer.
        opt.add_param_group({"params": [p2], "lr": 0.02})
        out = opt.state_dict()
        self.assertEqual(seen, [(2, [0.1, 0.02])],
                         "the hook must see the live optimizer with 2 groups")
        # The packed output reflects the same two groups (post-hoc view check).
        self.assertEqual(len(out["param_groups"]), 2)
        self.assertEqual(out["param_groups"][1]["lr"], 0.02)

    # ------------------------------------------------------------------
    # Return value is ignored; in-place side effects are visible
    # ------------------------------------------------------------------

    def test_pre_hook_return_value_ignored(self):
        """A returned dict is ignored: state_dict() still returns the packed dict."""
        opt, _ = self._make_optimizer(lr=0.1)
        bogus = {"bogus": True}
        opt.register_state_dict_pre_hook(lambda o: bogus)
        out = opt.state_dict()
        # The packed dict is returned regardless of the hook's return value.
        self.assertIn("state", out)
        self.assertIn("param_groups", out)
        self.assertNotIn("bogus", out, "the hook's return value must be ignored")
        self.assertEqual(out["param_groups"][0]["lr"], 0.1)
        self.assertEqual(out["param_groups"][0]["params"], [0])

    def test_pre_hook_inplace_side_effect_visible(self):
        """Side effects on the live optimizer are reflected in the packed output."""
        opt, _ = self._make_optimizer(lr=0.1)

        def bump_live(o):
            o.param_groups[0]["lr"] = 0.5
            return None  # the return value is ignored anyway

        opt.register_state_dict_pre_hook(bump_live)
        out = opt.state_dict()
        self.assertEqual(out["param_groups"][0]["lr"], 0.5,
                         "the pre-hook edit must be reflected in the packed output")
        self.assertEqual(opt.param_groups[0]["lr"], 0.5,
                         "the side effect must persist on the live optimizer")

    # ------------------------------------------------------------------
    # Ordering: prepend=True vs default registration order (two hooks)
    # ------------------------------------------------------------------

    def test_order_prepend_true_fires_new_hook_first(self):
        """prepend=True puts the newly registered hook before existing ones: [B, A]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))
        opt.register_state_dict_pre_hook(lambda o: order.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must fire B before the previously registered A")

    def test_order_default_follows_registration_order(self):
        """Default (prepend omitted/False) fires hooks in registration order: [A, B]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_state_dict_pre_hook(lambda o: order.append("A"))
        opt.register_state_dict_pre_hook(lambda o: order.append("B"), prepend=False)
        opt.state_dict()
        self.assertEqual(order, ["A", "B"],
                         "without prepend the registration order must be kept")

    # ------------------------------------------------------------------
    # Removal semantics
    # ------------------------------------------------------------------

    def test_remove_stops_firing(self):
        """A removed handle never fires again on subsequent state_dict() calls."""
        opt, _ = self._make_optimizer()
        calls = []
        h = opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_remove_one_of_two_keeps_other(self):
        """Removing one of two hooks leaves the other fully functional."""
        opt, _ = self._make_optimizer()
        c1, c2 = [], []
        h1 = opt.register_state_dict_pre_hook(lambda o: c1.append(1))
        opt.register_state_dict_pre_hook(lambda o: c2.append(1))
        opt.state_dict()
        self.assertEqual((len(c1), len(c2)), (1, 1))
        h1.remove()
        opt.state_dict()
        self.assertEqual((len(c1), len(c2)), (1, 2),
                         "removed hook stays silent while the kept hook still fires")

    # ------------------------------------------------------------------
    # Non-callable hooks
    # ------------------------------------------------------------------

    def test_non_callable_hook_raises_on_fire(self):
        """Non-callable hooks register without error but raise TypeError when fired."""
        opt, _ = self._make_optimizer()
        for bad in (123, None, "not-callable"):
            opt.register_state_dict_pre_hook(bad)
        with self.assertRaises(TypeError):
            opt.state_dict()

    # ------------------------------------------------------------------
    # Handle type
    # ------------------------------------------------------------------

    def test_handle_type(self):
        """Registration returns a RemovableHandle bound to this optimizer."""
        opt, _ = self._make_optimizer()
        handle = opt.register_state_dict_pre_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
