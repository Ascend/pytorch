# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.sgd.SGD.register_load_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.sgd.SGD.register_load_state_dict_post_hook
API 签名：register_load_state_dict_post_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后每次 load_state_dict() 触发一次；不注册则直接加载 | 已覆盖（test_hook_trigger/test_hook_fires_after_load） |
| 枚举选项 | prepend=False/True 次序：[B,A]/[A,B] | 已覆盖（hook_order_prepend_true_fires_first/hook_order_registration_order） |
| 参数类型 | hook(optimizer) 单实参回调，实参为该优化器本身（assertIs） | 已覆盖（test_hook_trigger） |
| 传参与不传参 | 显式 prepend=True 与默认（不传 prepend）调用 | 已覆盖（test_hook_trigger/test_hook_order_prepend_true_fires_first） |
| 等价类/边界值 | 触发位置在 load_state_dict 完成之后：hook 内可见已加载的 lr 与 SGD momentum_buffer 状态张量 | 已覆盖（test_hook_fires_after_load） |
| 正常传参场景 | 每次 load 触发一次；返回 RemovableHandle | 已覆盖（hook_trigger/hook_fires_after_load/hook_no_state_device_independent） |
| 异常传参场景 | 非 callable 注册不报错、触发时抛 TypeError；移除后不再触发；二删一留（删 A 留 B） | 已覆盖（test_hook_trigger/test_hook_two_removed_one_keeps） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移；优化器建于当前设备 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 混合设备类型：本接口为优化器回调注册的纯框架行为，hook 只收 optimizer 实例，无张量跨设备迁移语义；
  优化器参数与 momentum 状态建于当前设备（self.device_name），设备路径由 test_hook_fires_after_load 一并执行。
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
class TestSGDregister_load_state_dict_post_hook(TestCase):
    """Comprehensive cases for torch.optim.sgd.SGD.register_load_state_dict_post_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: the single-argument contract
    (this optimizer), dual-hook ordering for prepend=True/False, the
    fires-after-load position (the loaded lr and the SGD momentum_buffer
    state are already observable inside the hook), removal via
    RemovableHandle, remove-one-keep-one semantics, and the non-callable
    TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01, momentum=0.9):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.SGD([p], lr=lr, momentum=momentum)

    def _make_sgd_with_state(self, lr=0.1, momentum=0.9):
        """Run one real SGD step so momentum_buffer state exists on device."""
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0], device=self.device))
        opt = torch.optim.SGD([p], lr=lr, momentum=momentum)
        p.grad = torch.ones_like(p)
        opt.step()
        return opt, p

    def test_hook_trigger(self):
        """Hook is called at the documented lifecycle point."""
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
        self.assertEqual(len(calls2), 1)

        opt.register_load_state_dict_post_hook(3.14)
        with self.assertRaises(TypeError):
            opt.load_state_dict(opt.state_dict())

    def test_hook_order_prepend_true_fires_first(self):
        """A prepend=True hook registered later fires before the earlier one."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))  # prepend=False
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"],
                         "prepend=True must move the hook to the front (B then A)")

    def test_hook_order_registration_order(self):
        """Without prepend, hooks fire in registration order (A then B)."""
        order = []
        opt = self._make_optimizer()
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["A", "B"],
                         "default prepend=False must keep registration order")

    def test_hook_fires_after_load(self):
        """At fire time the loaded lr and momentum state are already applied."""
        opt, p = self._make_sgd_with_state(lr=0.1, momentum=0.9)
        sd = opt.state_dict()
        sd["param_groups"][0]["lr"] = 0.55  # the value load_state_dict must apply
        fired = []
        seen = {}

        def observe(o):
            fired.append(1)
            seen["lr"] = o.param_groups[0]["lr"]
            seen["momentum"] = o.param_groups[0]["momentum"]
            seen["state_keys"] = sorted(o.state[p].keys())
            seen["buffer"] = o.state[p].get("momentum_buffer")

        opt.register_load_state_dict_post_hook(observe)
        opt.load_state_dict(sd)
        self.assertEqual(len(fired), 1)
        self.assertEqual(seen["lr"], 0.55,
                         "post hook must observe the lr loaded by load_state_dict, "
                         "i.e. it fires after the load completes")
        self.assertEqual(seen["momentum"], 0.9)
        self.assertIn("momentum_buffer", seen["state_keys"],
                      "loaded SGD momentum state must be visible inside the post hook")
        self.assertIsInstance(seen["buffer"], torch.Tensor)
        # Each load_state_dict call fires the hook exactly once.
        opt.load_state_dict(sd)
        self.assertEqual(len(fired), 2)

    def test_hook_two_removed_one_keeps(self):
        """Removing one of two hooks keeps the other one firing."""
        opt = self._make_optimizer()
        calls_a, calls_b = [], []
        h_a = opt.register_load_state_dict_post_hook(lambda o: calls_a.append(1))
        h_b = opt.register_load_state_dict_post_hook(lambda o: calls_b.append(1))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual((len(calls_a), len(calls_b)), (1, 1))
        h_a.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls_a), 1, "removed hook must not fire again")
        self.assertEqual(len(calls_b), 2, "the kept hook must keep firing")
        h_b.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual((len(calls_a), len(calls_b)), (1, 2))

    def test_hook_no_state_device_independent(self):
        """Optimizer hook registration is pure framework state; device-agnostic."""
        # No tensor is moved; a fresh optimizer exercises the same hook contract
        # on the current device without relying on a device kernel.
        opt = self._make_optimizer()
        handle = opt.register_load_state_dict_post_hook(lambda *a, **k: None)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        handle.remove()


if __name__ == "__main__":
    run_tests()
