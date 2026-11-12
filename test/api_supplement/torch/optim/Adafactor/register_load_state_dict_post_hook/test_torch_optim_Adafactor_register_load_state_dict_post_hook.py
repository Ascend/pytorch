# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.Adafactor.register_load_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.Adafactor.register_load_state_dict_post_hook
API 签名：register_load_state_dict_post_hook(self, hook: Callable[[Optimizer], None], prepend: bool = False) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 hook 后 load_state_dict() 触发；不注册则不触发 | 已覆盖（_fires_with_optimizer_argument/_removed_does_not_fire） |
| 枚举选项 | prepend=False/True 控制触发次序（双 hook [A,B]/[B,A]） | 已覆盖（_order_default_registration/_order_prepend） |
| 参数类型 | hook(opt) 单参回调；返回 RemovableHandle | 已覆盖（_fires_with_optimizer_argument/_removed_does_not_fire） |
| 传参与不传参 | 显式 hook/prepend 与默认调用 | 已覆盖（_order_default_registration/_order_prepend） |
| 等价类/边界值 | 触发位置在 load_state_dict 完成之后（hook 可观测已载入的 state 数值） | 已覆盖（_fires_after_state_is_applied） |
| 正常传参场景 | 注册后在 load_state_dict() 完成后确实触发且实参为优化器实例 | 已覆盖（_fires_with_optimizer_argument） |
| 异常传参场景 | 非 callable hook：注册成功、load 时抛 TypeError | 已覆盖（_non_callable_raises_on_load） |
| 混合设备类型 | 纯框架回调注册，不涉及张量迁移 | 未覆盖（无设备对照语义；state 观测用例在当前设备上执行） |

未覆盖项及原因：
- 无（hook 为纯 Python 回调注册，无张量数值语义；实测确认注册非 callable 不会立刻报错，
  TypeError 在 load_state_dict 调用 hook 时抛出，消息为 "'float' object is not callable"）。
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
class TestAdafactorregister_load_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.Adafactor.register_load_state_dict_post_hook (NPU/CUDA portable).

    Adafactor inherits this hook registry from Optimizer, but the file still
    exercises the full contract independently on an Adafactor instance:
    argument identity, firing point (after the loaded state is applied, with
    the Adafactor-specific factored state observable), dual-hook ordering for
    both prepend values, removal and non-callable error timing.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_optimizer(self):
        p = torch.nn.Parameter(torch.randn(3, device=self.device), requires_grad=True)
        return torch.optim.Adafactor([p], lr=0.01)

    def _stepped_isolated_state_dict(self, opt):
        """Step once and return a deepcopy'd state_dict (live-reference safe)."""
        p = opt.param_groups[0]["params"][0]
        p.grad = torch.full_like(p, 0.5)
        opt.step()
        return copy.deepcopy(opt.state_dict())

    def test_post_hook_fires_with_optimizer_argument(self):
        """Hook is invoked once with the optimizer instance as its sole argument."""
        opt = self._make_optimizer()
        calls = []
        opt.register_load_state_dict_post_hook(lambda o: calls.append(o))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1, "hook must fire exactly once per load")
        # The hook argument must be the optimizer instance itself.
        self.assertIs(calls[0], opt)

    def test_post_hook_fires_after_state_is_applied(self):
        """A post hook observes the loaded Adafactor state, proving post-load timing."""
        src = self._make_optimizer()
        sd = self._stepped_isolated_state_dict(src)  # step == 1, variance == 0.25
        dst = self._make_optimizer()
        p_dst = dst.param_groups[0]["params"][0]
        observed = []
        dst.register_load_state_dict_post_hook(
            lambda o: observed.append(
                (o.state[p_dst]["step"].item(),
                 o.state[p_dst]["variance"].mean().item())))
        dst.load_state_dict(sd)
        # If the hook ran BEFORE the state was applied, it would observe the
        # fresh optimizer's empty state (KeyError) instead of the loaded
        # Adafactor values (step=1.0, variance mean=0.25).
        self.assertEqual(observed, [(1.0, 0.25)])

    def test_post_hook_order_default_registration(self):
        """prepend=False (default): hooks fire in registration order [A, B]."""
        opt = self._make_optimizer()
        order = []
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"))
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["A", "B"])

    def test_post_hook_order_prepend(self):
        """prepend=True: the later-registered hook fires first, giving [B, A]."""
        opt = self._make_optimizer()
        order = []
        opt.register_load_state_dict_post_hook(lambda o: order.append("A"))
        opt.register_load_state_dict_post_hook(lambda o: order.append("B"), prepend=True)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(order, ["B", "A"])

    def test_post_hook_removed_does_not_fire(self):
        """After handle.remove(), the hook is not invoked by further loads."""
        opt = self._make_optimizer()
        calls = []
        handle = opt.register_load_state_dict_post_hook(lambda o: calls.append(o))
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1)
        handle.remove()
        opt.load_state_dict(opt.state_dict())
        self.assertEqual(len(calls), 1, "removed hook must not fire again")

    def test_post_hook_non_callable_raises_on_load(self):
        """Registering a non-callable succeeds; TypeError surfaces at load time."""
        opt = self._make_optimizer()
        # Verified behavior: registration itself does not validate callability
        # and returns a valid handle; the error is raised when load_state_dict
        # tries to invoke the hook.
        handle = opt.register_load_state_dict_post_hook(3.14)
        self.assertIsInstance(handle, torch.utils.hooks.RemovableHandle)
        with self.assertRaisesRegex(TypeError, "object is not callable"):
            opt.load_state_dict(opt.state_dict())
        handle.remove()


if __name__ == "__main__":
    run_tests()
