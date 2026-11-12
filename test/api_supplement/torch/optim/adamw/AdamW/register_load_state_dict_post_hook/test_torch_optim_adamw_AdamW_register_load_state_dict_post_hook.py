# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adamw.AdamW.register_load_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.adamw.AdamW.register_load_state_dict_post_hook
API 签名：register_load_state_dict_post_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空优化器与 step 产生非空 state 后均触发 | 已覆盖（test_hook_single_arg_contract / test_post_hook_fires_after_load） |
| 枚举选项 | prepend=True/False 双次序 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 参数类型 | hook 单实参回调；非 callable 触发 TypeError | 已覆盖（hook_single_arg_contract/non_callable_hook_raises_on_fire） |
| 传参与不传参 | 显式 prepend=True 与省略 | 已覆盖（order_prepend_true_fires_new_hook_first/order_default_follows_registration_order） |
| 等价类/边界值 | 触发点在加载完成后：hook 内可见已加载 lr 与 state（而非加载前旧值） | 已覆盖（test_post_hook_fires_after_load） |
| 正常传参场景 | 每次 load_state_dict 恰好触发一次；返回 RemovableHandle | 已覆盖（test_hook_fires_once_per_load / test_handle_type） |
| 异常传参场景 | 非 callable hook 注册不抛错、触发时抛 TypeError | 已覆盖（test_non_callable_hook_raises_on_fire） |
| 移除语义 | remove 后不再触发；二删一留 | 已覆盖（test_remove_stops_firing / test_remove_one_of_two_keeps_other） |
| 混合设备类型 | hook 内可见已加载到当前设备（NPU/CUDA）的 state 张量 | 已覆盖（test_post_hook_fires_after_load 断言 state 张量 device.type 与当前设备一致） |

未覆盖项及原因：
- hook 返回值分支：本 hook 契约为 hook(optimizer) -> None，Optimizer.load_state_dict 仅以
  post_hook(self) 调用且不消费返回值，无返回值语义可测。
- 本接口为纯框架回调注册，无数值执行语义；state 张量的设备断言在 test_post_hook_fires_after_load 内完成。
"""
import copy
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
class TestAdamWregister_load_state_dict_post_hook(TestCase):
    """Comprehensive cases for torch.optim.adamw.AdamW.register_load_state_dict_post_hook (NPU/CUDA portable).

    AdamW inherits the hook machinery from Optimizer, but this file still covers
    the full contract independently: the single-positional-arg contract, the
    fires-after-load timing (loaded hyperparams/state visible inside the hook),
    prepend ordering with two hooks ([B, A] vs [A, B]), removal semantics and
    the non-callable TypeError-on-fire path.
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
    # Argument contract and firing point
    # ------------------------------------------------------------------

    def test_hook_single_arg_contract(self):
        """The hook receives exactly one positional argument: the optimizer itself."""
        opt, _ = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_post_hook(lambda *a, **k: calls.append((a, k)))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(len(args), 1, "the hook must receive a single positional arg")
        self.assertIs(args[0], opt, "the positional arg must be the optimizer itself")
        self.assertEqual(kwargs, {}, "no keyword arguments are passed to the hook")

    def test_post_hook_fires_after_load(self):
        """Inside the hook the LOADED lr/state are already visible (fires_after_load)."""
        opt, p = self._make_stepped_optimizer(lr=0.1)
        sd = copy.deepcopy(opt.state_dict())
        # Diverge the live optimizer so a stale pre-load view would be caught.
        opt.param_groups[0]["lr"] = 99.0
        snapshot = {}

        def hook(o):
            st = o.state[o.param_groups[0]["params"][0]]
            snapshot["lr"] = o.param_groups[0]["lr"]
            snapshot["state_keys"] = sorted(st.keys())
            snapshot["step"] = float(st["step"])
            snapshot["state_device"] = st["exp_avg"].device.type

        opt.register_load_state_dict_post_hook(hook)
        opt.load_state_dict(sd)
        # The hook fired AFTER loading: it must see the loaded lr (0.1), not 99.0.
        self.assertEqual(snapshot["lr"], 0.1,
                         "hook must observe the loaded lr, not the pre-load value")
        # AdamW state (step/exp_avg/exp_avg_sq) was restored and observable inside.
        self.assertEqual(snapshot["state_keys"], ["exp_avg", "exp_avg_sq", "step"])
        self.assertEqual(snapshot["step"], 1.0, "loaded step counter must be 1")
        # The restored state tensors live on the param's device (NPU/CUDA/CPU).
        self.assertEqual(snapshot["state_device"], self.device.type)

    def test_hook_fires_once_per_load(self):
        """Each load_state_dict call fires the hook exactly once."""
        opt, _ = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_post_hook(lambda o: calls.append(o))
        sd = opt.state_dict()
        opt.load_state_dict(sd)
        self.assertEqual(len(calls), 1)
        opt.load_state_dict(sd)
        self.assertEqual(len(calls), 2, "a second load must fire the hook a second time")

    # ------------------------------------------------------------------
    # Ordering: prepend=True vs default registration order (two hooks)
    # ------------------------------------------------------------------

    def test_order_prepend_true_fires_new_hook_first(self):
        """prepend=True puts the newly registered hook before existing ones: [B, A]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must fire B before the previously registered A")

    def test_order_default_follows_registration_order(self):
        """Default (prepend omitted/False) fires hooks in registration order: [A, B]."""
        opt, _ = self._make_optimizer()
        order = []
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"), prepend=False)
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
        h = opt.register_load_state_dict_post_hook(lambda o: calls.append(o))
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
        h1 = opt.register_load_state_dict_post_hook(lambda o: c1.append(1))
        opt.register_load_state_dict_post_hook(lambda o: c2.append(1))
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
        sd = opt.state_dict()
        for bad in (3.14, None, "not-callable"):
            opt.register_load_state_dict_post_hook(bad)
        with self.assertRaises(TypeError):
            opt.load_state_dict(sd)

    # ------------------------------------------------------------------
    # Handle type / device-agnostic registration
    # ------------------------------------------------------------------

    def test_handle_type(self):
        """Registration returns a RemovableHandle bound to this optimizer."""
        opt, _ = self._make_optimizer()
        handle = opt.register_load_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
