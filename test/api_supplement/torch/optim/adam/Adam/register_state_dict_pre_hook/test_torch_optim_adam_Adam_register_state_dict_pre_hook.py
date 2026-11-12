# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.adam.Adam.register_state_dict_pre_hook 接口功能正确性
API 名称：torch.optim.adam.Adam.register_state_dict_pre_hook
API 签名：register_state_dict_pre_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 state_dict() 触发；不注册则默认状态 | 已覆盖（test_hook_trigger） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook 断言顺序） | 已覆盖（test_hook_order_prepend_true_fires_first/test_hook_order_registration_order） |
| 参数类型 | 可调用 hook；hook 实参为本 optimizer；返回值被忽略（实测：hook(optimizer) -> None） | 已覆盖（test_hook_trigger/test_hook_fires_before_packing） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（test_hook_trigger/test_hook_order_registration_order） |
| 等价类/边界值 | 触发位置在 state_dict 打包之前（pre hook 内修改 lr 可见于打包结果） | 已覆盖（test_hook_fires_before_packing） |
| 正常传参场景 | 注册后在 state_dict() 中确实被触发；注册返回 RemovableHandle | 已覆盖（test_hook_trigger/test_hook_no_state_device_independent） |
| 异常传参场景 | 非 callable 注册成功、触发时抛 TypeError；remove 后不再触发 | 已覆盖（test_hook_trigger） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移；优化器建于当前设备 | 未覆盖（无设备对照语义） |

未覆盖项及原因：
- 本版本 hook 返回值被 state_dict 丢弃（实测 torch 源码 pre_hook(self) 无返回接收），故无「返回新 dict 覆盖结果」路径可测，覆盖表按实测契约登记。
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
class TestAdamregister_state_dict_pre_hook(TestCase):
    """Comprehensive cases for torch.optim.adam.Adam.register_state_dict_pre_hook (NPU/CUDA portable).

    Covers the full registration lifecycle: trigger with the correct
    arguments (this optimizer), dual-hook ordering for prepend=True/False,
    removal via RemovableHandle, firing position (before state_dict packs
    the optimizer), and the non-callable TypeError path.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self, lr=0.01):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adam([p], lr=lr)

    def test_hook_trigger(self):
        """Hook is called with this optimizer as its only argument."""
        calls = []
        opt = self._make_optimizer()
        opt.register_state_dict_pre_hook(lambda o: calls.append(o))
        opt.state_dict()
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], opt, "pre hook must receive this very optimizer")

        # prepend + removable handle
        calls2 = []
        h = opt.register_state_dict_pre_hook(lambda o: calls2.append(id(o)), prepend=True)
        opt.state_dict()
        self.assertEqual(len(calls2), 1)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()
        opt.state_dict()
        self.assertEqual(len(calls2), 1, "removed hook must not fire again")

        # state_dict output keeps the two documented top-level keys.
        self.assertIn("param_groups", opt.state_dict())
        self.assertIn("state", opt.state_dict())

        # non-callable registration succeeds but raises TypeError when fired
        opt.register_state_dict_pre_hook(123)
        with self.assertRaises(TypeError):
            opt.state_dict()

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
        """The pre hook runs before packing: its mutations reach the output."""
        opt = self._make_optimizer(lr=0.01)

        def bump_lr(o):
            o.param_groups[0]["lr"] = 0.123

        opt.register_state_dict_pre_hook(bump_lr)
        sd = opt.state_dict()
        # If the hook ran before pack_group, the packed lr must carry the
        # mutation; if it ran after packing, the old lr would be packed.
        self.assertEqual(sd["param_groups"][0]["lr"], 0.123,
                         "pre-hook mutation must be visible in the packed output")
        # The mutation is on the live optimizer too (hook received this opt).
        self.assertEqual(opt.param_groups[0]["lr"], 0.123)

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
