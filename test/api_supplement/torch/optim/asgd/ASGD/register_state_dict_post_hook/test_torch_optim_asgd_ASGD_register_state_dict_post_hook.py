# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.asgd.ASGD.register_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.asgd.ASGD.register_state_dict_post_hook
API 签名：register_state_dict_post_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空优化器与 step 产生非空 state 后均触发 | 已覆盖（test_post_hook_two_arg_contract / test_post_hook_fires_after_packing） |
| 枚举选项 | prepend=True/False 双次序 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 参数类型 | hook 双实参回调；非 callable 触发 TypeError | 已覆盖（post_hook_two_arg_contract/non_callable_hook_raises_on_fire） |
| 传参与不传参 | 显式 prepend=True 与省略 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 等价类/边界值 | 打包后触发：params/state 键为 int；返回 dict 替换、None 保留；in-place（含 lr）可见 | 已覆盖（post_hook_fires_after_packing/
post_hook_return_dict_replaces_output/post_hook_return_none_keeps_packed_output/post_hook_inplace_edit_visible） |
| 正常传参场景 | 注册 → state_dict() 触发一次；返回 RemovableHandle | 已覆盖（test_post_hook_two_arg_contract / test_remove_stops_firing） |
| 异常传参场景 | 非 callable hook 注册不抛错、触发时抛 TypeError | 已覆盖（test_non_callable_hook_raises_on_fire） |
| 移除语义 | remove 后不再触发；二删一留 | 已覆盖（test_remove_stops_firing / test_remove_one_of_two_keeps_other） |
| 混合设备类型 | 纯框架回调注册，参数张量位于当前设备（NPU/CUDA） | 已覆盖（全部用例在 self.device 上构造参数） |

未覆盖项及原因：
- 返回 dict 的深层结构约束：state_dict() 消费 hook 返回值时只要求非 None，不校验结构；
  结构合法性由 load_state_dict 用例覆盖，本文件只测输出替换契约。
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
class TestASGDregister_state_dict_post_hook(TestCase):
    """Comprehensive cases for torch.optim.asgd.ASGD.register_state_dict_post_hook (NPU/CUDA portable).

    ASGD inherits the hook machinery from Optimizer, but this file still covers
    the full contract independently: the two-positional-arg contract
    (optimizer, packed state dict), the fires-after-packing timing (params are
    int indices, state keys are ints), the return-dict path (the return value
    replaces the output), the return-None path, the in-place edit path
    (including an lr edit), prepend ordering with two hooks ([B, A] vs [A, B]),
    removal semantics and the non-callable TypeError-on-fire path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.05):
        p = torch.nn.Parameter(torch.ones(3, device=self.device), requires_grad=True)
        return torch.optim.ASGD([p], lr=lr), p

    def _make_stepped_optimizer(self, lr=0.05):
        """Step once so the packed state dict carries real entries."""
        opt, p = self._make_optimizer(lr)
        p.grad = torch.ones_like(p)
        opt.step()
        return opt, p

    # ------------------------------------------------------------------
    # Argument contract and firing point
    # ------------------------------------------------------------------

    def test_post_hook_two_arg_contract(self):
        """The hook receives exactly (optimizer, packed_state_dict)."""
        opt, _ = self._make_optimizer()
        calls = []
        opt.register_state_dict_post_hook(lambda *a, **k: calls.append((a, k)))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(len(args), 2, "the hook must receive two positional args")
        self.assertIs(args[0], opt, "first positional arg must be the optimizer itself")
        self.assertIsInstance(args[1], dict, "second positional arg must be the packed dict")
        self.assertEqual(kwargs, {}, "no keyword arguments are passed to the hook")

    def test_post_hook_fires_after_packing(self):
        """The hook sees the packed dict: params are int indices, state keys are ints."""
        opt, _ = self._make_stepped_optimizer(lr=0.05)
        seen = {}

        def hook(o, sd):
            seen["params"] = sd["param_groups"][0]["params"]
            seen["state_keys"] = list(sd["state"].keys())
            return None

        opt.register_state_dict_post_hook(hook)
        out = opt.state_dict()
        # Params are packed as int order indices, NOT Tensor objects.
        self.assertEqual(seen["params"], [0])
        self.assertTrue(all(isinstance(i, int) for i in seen["params"]),
                        "packed params must be int indices, not Tensors")
        # State is remapped to int indices as well.
        self.assertEqual(seen["state_keys"], [0])
        self.assertTrue(all(isinstance(k, int) for k in seen["state_keys"]),
                        "packed state keys must be int indices")
        # The same packed dict is what state_dict() returns to the caller.
        self.assertEqual(out["param_groups"][0]["params"], [0])

    # ------------------------------------------------------------------
    # Output manipulation: return-replaces, return-None, in-place edit
    # ------------------------------------------------------------------

    def test_post_hook_return_dict_replaces_output(self):
        """A returned dict replaces the state_dict() output entirely."""
        opt, _ = self._make_optimizer()
        opt.register_state_dict_post_hook(lambda o, sd: {"fresh": True})
        out = opt.state_dict()
        self.assertEqual(out, {"fresh": True},
                         "the hook's returned dict must be the state_dict() output")

    def test_post_hook_return_none_keeps_packed_output(self):
        """Returning None keeps the packed dict as the output."""
        opt, _ = self._make_optimizer(lr=0.05)
        opt.register_state_dict_post_hook(lambda o, sd: None)
        out = opt.state_dict()
        self.assertIn("state", out)
        self.assertIn("param_groups", out)
        self.assertEqual(out["param_groups"][0]["lr"], 0.05)
        self.assertEqual(out["param_groups"][0]["params"], [0])

    def test_post_hook_inplace_edit_visible(self):
        """In-place edits (with no return) are visible in the returned dict."""
        opt, _ = self._make_optimizer(lr=0.05)

        def bump(o, sd):
            sd["param_groups"][0]["lr"] = 0.5
            sd["extra"] = 42
            return None  # returning None must NOT undo the in-place edits

        opt.register_state_dict_post_hook(bump)
        out = opt.state_dict()
        self.assertEqual(out["param_groups"][0]["lr"], 0.5,
                         "in-place lr edit must be visible in the output")
        self.assertEqual(out["extra"], 42,
                         "in-place key injection must be visible in the output")

    def test_post_hook_returning_same_dict_injects_keys(self):
        """Returning the (mutated) dict itself is consumed as the output too."""
        opt, _ = self._make_optimizer()

        def inject(o, sd):
            sd["extra"] = 42
            return sd

        opt.register_state_dict_post_hook(inject)
        out = opt.state_dict()
        self.assertEqual(out.get("extra"), 42)

    # ------------------------------------------------------------------
    # Ordering: prepend=True vs default registration order (two hooks)
    # ------------------------------------------------------------------

    def test_order_prepend_true_fires_new_hook_first(self):
        """prepend=True puts the newly registered hook before existing ones: [B, A]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_state_dict_post_hook(lambda o, sd: order.append("A"))
        opt.register_state_dict_post_hook(lambda o, sd: order.append("B"), prepend=True)
        opt.state_dict()
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must fire B before the previously registered A")

    def test_order_default_follows_registration_order(self):
        """Default (prepend omitted/False) fires hooks in registration order: [A, B]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_state_dict_post_hook(lambda o, sd: order.append("A"))
        opt.register_state_dict_post_hook(lambda o, sd: order.append("B"), prepend=False)
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
        h = opt.register_state_dict_post_hook(lambda o, sd: calls.append(1))
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
        h1 = opt.register_state_dict_post_hook(lambda o, sd: c1.append(1))
        opt.register_state_dict_post_hook(lambda o, sd: c2.append(1))
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
        for bad in (3.14, None, "not-callable"):
            opt.register_state_dict_post_hook(bad)
        with self.assertRaises(TypeError):
            opt.state_dict()

    # ------------------------------------------------------------------
    # Handle type
    # ------------------------------------------------------------------

    def test_handle_type(self):
        """Registration returns a RemovableHandle bound to this optimizer."""
        opt, _ = self._make_optimizer()
        handle = opt.register_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
