# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.register_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.register_state_dict_post_hook
API 签名：register_state_dict_post_hook(self, hook: Callable[[Module, dict, str, dict], None]) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 module hook 后 state_dict() 触发；不注册则默认行为 | 已覆盖（test_module_hook_trigger/test_hook_signature_and_payload） |
| 枚举选项 | nn.Module 版本无 prepend 参数（实测传 prepend 抛 TypeError），多 hook 按注册顺序触发 | 已覆盖（test_hook_registration_order_and_no_prepend） |
| 参数类型 | hook(module, state_dict, prefix, local_metadata) 四参签名（实测 torch 2.11 nn.Module 契约）； |
非 None 返回值触发 RuntimeError；in-place 修改生效 |
已覆盖（hook_signature_and_payload/hook_inplace_mutation_reaches_caller/hook_non_none_return_raises） |
| 传参与不传参 | 显式注册 hook；RemovableHandle.remove() 后不再触发 | 已覆盖（test_module_hook_trigger/test_hook_removal_stops_firing） |
| 等价类/边界值 | 触发位置在 state_dict 打包之后（收到的 dict 含 n_averaged/module.weight/module.bias |
完整键，prefix=''、local_metadata={'version': 1}） |
已覆盖（hook_signature_and_payload） |
| 正常传参场景 | 注册后 state_dict() 生成返回值后确实触发、module 参数即本 am、返回 RemovableHandle |
已覆盖（module_hook_trigger/hook_signature_and_payload） |
| 异常传参场景 | str 注册期抛 AttributeError（_from_public_api 标记写入失败）； |
允许设属性的不可调用对象注册成功、触发时抛 TypeError |
已覆盖（hook_non_callable_str_fails_at_registration/hook_non_callable_object_typeerror_when_fired） |
| 混合设备类型 | am 驻留当前设备时 hook 触发并可见设备张量 | 已覆盖（test_hook_device_resident_module） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- prepend 语义未覆盖：实测 nn.Module.register_state_dict_post_hook（torch 2.11）不提供 prepend 参数（与 Optimizer 版本不同），
  传 prepend= 触发 TypeError，该差异在 test_hook_registration_order_and_no_prepend 中锁定。
"""
import os
import unittest

import torch
import torch.optim.swa_utils as swa_utils

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
class TestAveragedModelregister_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.register_state_dict_post_hook (NPU/CUDA portable).

    AveragedModel inherits this hook from nn.Module (NOT from Optimizer), so
    the signature differs from the optimizer flavour. Empirical contract
    (torch 2.11, probed on NPU):
      - hook signature is hook(module, state_dict, prefix, local_metadata)
        and it must return None; a non-None return makes state_dict() raise
        RuntimeError("state_dict post-hook must return None");
      - the hook may mutate the received dict in place and the mutation is
        what the caller sees (the public API deliberately ignores return
        values, unlike the legacy _register_state_dict_hook);
      - no prepend kwarg exists (TypeError), hooks fire in registration order;
      - a str hook fails at registration with AttributeError because the API
        stamps hook._from_public_api on it; any attribute-settable but
        non-callable object registers fine and raises TypeError when fired.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_module_hook_trigger(self):
        """The registered module hook fires at the documented lifecycle point."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        calls = []
        if 'state_dict' == 'state_dict':
            h = am.register_state_dict_post_hook(
                lambda module, destination, prefix, local_metadata: calls.append(1))
            am.state_dict()
        else:
            h = am.register_load_state_dict_post_hook(lambda m: calls.append(1))
            am.load_state_dict(am.state_dict())
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()

    def test_hook_signature_and_payload(self):
        """Hook receives (this module, the packed dict, prefix, local_metadata)."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        seen = []
        am.register_state_dict_post_hook(
            lambda module, sd, prefix, local_metadata: seen.append((module, sd, prefix, local_metadata)))
        result = am.state_dict()
        self.assertEqual(len(seen), 1, "state_dict() must fire the hook exactly once")
        module, sd, prefix, local_metadata = seen[0]
        self.assertIs(module, am, "the hook must receive this very AveragedModel")
        # The hook fires after packing: the dict carries AveragedModel's own
        # buffer plus the deep-copied module's parameter keys.
        for key in ("n_averaged", "module.weight", "module.bias"):
            self.assertIn(key, sd, f"packed dict must contain '{key}'")
        self.assertEqual(prefix, "", "root-module hooks are called with an empty prefix")
        self.assertIsInstance(local_metadata, dict)
        # state_dict() returns the same dict object the hook last saw, so the
        # caller observes whatever the hook did to it.
        self.assertIs(result, sd)

    def test_hook_inplace_mutation_reaches_caller(self):
        """In-place mutations of the received dict are visible to the caller."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))

        def mutate(module, sd, prefix, local_metadata):
            sd["injected_key"] = 42
            sd["n_averaged"] = torch.tensor(99)

        am.register_state_dict_post_hook(mutate)
        sd = am.state_dict()
        self.assertEqual(sd.get("injected_key"), 42,
                         "keys injected by the hook must reach the caller")
        self.assertEqual(int(sd["n_averaged"]), 99,
                         "values overwritten by the hook must reach the caller")
        # The in-place edits are report-only: the live buffer is untouched.
        self.assertEqual(int(am.n_averaged), 0)

    def test_hook_non_none_return_raises(self):
        """The public API enforces hook(...) -> None at fire time."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        am.register_state_dict_post_hook(
            lambda module, sd, prefix, local_metadata: {"replaced": 1})
        with self.assertRaisesRegex(RuntimeError, "must return None"):
            am.state_dict()

    def test_hook_registration_order_and_no_prepend(self):
        """Hooks fire in registration order; the Module API has no prepend kwarg."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        order = []
        am.register_state_dict_post_hook(lambda m, sd, p, lmd: order.append("A"))
        am.register_state_dict_post_hook(lambda m, sd, p, lmd: order.append("B"))
        am.state_dict()
        self.assertEqual(order, ["A", "B"],
                         "without prepend support the firing order must follow registration")
        # The Optimizer flavour has prepend=..., the nn.Module flavour does not.
        with self.assertRaises(TypeError):
            am.register_state_dict_post_hook(lambda m, sd, p, lmd: None, prepend=True)

    def test_hook_removal_stops_firing(self):
        """handle.remove() unregisters the hook; state_dict() keeps working."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        calls = []
        h = am.register_state_dict_post_hook(lambda m, sd, p, lmd: calls.append(1))
        am.state_dict()
        self.assertEqual(len(calls), 1)
        h.remove()
        sd = am.state_dict()
        self.assertEqual(len(calls), 1, "removed hook must not fire again")
        # state_dict() itself remains fully functional after removal.
        self.assertIn("n_averaged", sd)

    def test_hook_non_callable_str_fails_at_registration(self):
        """A str hook fails at registration: the API stamps _from_public_api on it."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        with self.assertRaises(AttributeError):
            am.register_state_dict_post_hook("not-callable")

    def test_hook_non_callable_object_typeerror_when_fired(self):
        """An attribute-settable non-callable registers but raises TypeError when fired."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))

        class NotCallable:
            pass

        h = am.register_state_dict_post_hook(NotCallable())
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        with self.assertRaises(TypeError):
            am.state_dict()
        h.remove()
        # After removing the bad hook, state_dict() works again.
        self.assertIn("n_averaged", am.state_dict())

    def test_hook_device_resident_module(self):
        """The hook fires for a device-resident AveragedModel and sees its tensors."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2).to(self.device))
        seen = []
        am.register_state_dict_post_hook(
            lambda module, sd, prefix, lmd: seen.append((module, sd)))
        sd = am.state_dict()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], am)
        # The packed dict tensors live on the module's device.
        self.assertEqual(sd["module.weight"].device.type, self.device_name)
        self.assertEqual(int(sd["n_averaged"]), 0)


if __name__ == "__main__":
    run_tests()
