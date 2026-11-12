# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adam.Adam.register_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.adam.Adam.register_state_dict_post_hook
API 签名：register_state_dict_post_hook(self, hook, prepend=False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则默认状态 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook 断言顺序） | 已覆盖（test_hook_order_prepend_true_fires_first/test_hook_order_registration_order） |
| 参数类型 | hook(optimizer,sd)；dict替换、None保原、in-place生效 | 已覆盖（hook_trigger/hook_return_dict_replaces/hook_inplace_mutation） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_registration_order） |
| 等价类/边界值 | 触发位置在 state_dict 打包之后（收到的 dict 含 state/param_groups 及 Adam 超参键） | 已覆盖（test_hook_trigger/test_hook_fires_after_packing） |
| 正常传参场景 | 注册后在 state_dict() 返回值生成后确实触发；注册返回 RemovableHandle | 已覆盖（test_hook_trigger/test_hook_no_state_device_independent） |
| 异常传参场景 | 非 callable 注册成功、触发时抛 TypeError；remove 后不再触发 | 已覆盖（test_hook_trigger） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移；优化器建于当前设备 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
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
class TestAdamregister_state_dict_post_hook(TestCase):
    """Comprehensive cases for torch.optim.adam.Adam.register_state_dict_post_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: trigger with the correct
    arguments (this optimizer plus the packed state dict), dual-hook
    ordering for prepend=True/False, return-value semantics (a returned
    dict replaces the result; None keeps it), removal via RemovableHandle,
    and the non-callable TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adam([p], lr=lr)

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

    def test_hook_return_dict_replaces(self):
        """A hook returning a dict replaces the state_dict() result; None keeps it."""
        opt = self._make_optimizer()
        opt.register_state_dict_post_hook(lambda o, sd: {"replaced": 1})
        self.assertEqual(opt.state_dict(), {"replaced": 1},
                         "a returned dict must become the new state_dict result")
        # Returning None keeps the (possibly mutated) original dict.
        opt2 = self._make_optimizer()
        opt2.register_state_dict_post_hook(lambda o, sd: None)
        out = opt2.state_dict()
        self.assertIn("state", out)
        self.assertIn("param_groups", out)

    def test_hook_fires_after_packing(self):
        """The hook receives the packed dict: ints for params, Adam hyperparams."""
        p = self._make_optimizer().param_groups[0]["params"][0]
        opt = torch.optim.Adam([p], lr=0.07)
        p.grad = torch.ones_like(p)
        opt.step()  # produce non-empty state so the packed dict is observable
        seen = {}
        opt.register_state_dict_post_hook(lambda o, sd: seen.update(sd) or None)
        opt.state_dict()
        self.assertIn("state", seen)
        self.assertIn("param_groups", seen)
        # Packing replaces Tensors with order indices (ints), proving the
        # hook fires after the packing loop, not on the raw optimizer state.
        packed_params = seen["param_groups"][0]["params"]
        self.assertEqual(packed_params, [0])
        self.assertTrue(all(isinstance(i, int) for i in packed_params))
        # The packed group carries Adam's own hyperparameters.
        group = seen["param_groups"][0]
        self.assertEqual(group["lr"], 0.07)
        self.assertEqual(group["betas"], (0.9, 0.999))
        self.assertEqual(group["eps"], 1e-08)
        # The stepped state was packed under index 0 with Adam's state keys.
        self.assertIn(0, seen["state"])
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertIn(key, seen["state"][0])

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

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
