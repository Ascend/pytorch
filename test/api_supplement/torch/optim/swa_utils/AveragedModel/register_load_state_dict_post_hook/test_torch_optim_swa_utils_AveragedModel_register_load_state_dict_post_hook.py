# -*- coding: utf-8 -*-
"""测试目的：验证 torch.optim.swa_utils.AveragedModel.register_load_state_dict_post_hook 接口功能正确性
API 名称：torch.optim.swa_utils.AveragedModel.register_load_state_dict_post_hook
API 签名：register_load_state_dict_post_hook(self, hook: Callable[[Module, _IncompatibleKeys], None]) -> RemovableHandle

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 注册 module hook 后 load_state_dict 触发；不注册则默认行为 |
已覆盖（module_hook_trigger/hook_signature_and_loaded_weights_visible） |
| 枚举选项 | nn.Module 版本无 prepend 参数（实测传 prepend 抛 TypeError），多 hook 按注册顺序触发 | 已覆盖（test_hook_registration_order_and_no_prepend） |
| 参数类型 | hook(module, incompatible_keys) 两参签名（实测 torch 2.11 nn.Module 契约）， |
incompatible_keys 为 _IncompatibleKeys(missing_keys, unexpected_keys) 且可被 hook 原地改写； |
hook 返回非 None 触发 AssertionError |
已覆盖（hook_signature_and_loaded_weights_visible/hook_sees_missing_and_unexpected_keys/ |
hook_can_clear_incompatible_keys/hook_non_none_return_raises） |
| 传参与不传参 | 显式注册 hook；RemovableHandle.remove() 后不再触发 | 已覆盖（test_module_hook_trigger/test_hook_removal_stops_firing） |
| 等价类/边界值 | 触发位置在加载完成之后（hook 内已可见新权重数值）；strict=False 下 missing/unexpected 键如实上报 |
已覆盖（hook_signature_and_loaded_weights_visible/hook_sees_missing_and_unexpected_keys） |
| 正常传参场景 | 加载后确实触发、module 参数即本 am、返回 RemovableHandle；hook 清空 incompatible_keys 可令 strict=True 通过 |
已覆盖（module_hook_trigger/hook_signature_and_loaded_weights_visible/hook_can_clear_incompatible_keys） |
| 异常传参场景 | 非 callable（str）注册成功、触发时抛 TypeError；传入非 state_dict 结构时 strict=True 抛 RuntimeError（与 hook 无关的对照） |
已覆盖（hook_non_callable_typeerror_when_fired/strict_load_still_validates_payload） |
| 混合设备类型 | am 驻留当前设备时加载设备张量并触发 hook | 已覆盖（test_hook_device_resident_load） |

未覆盖项及原因：
- 本接口本质为优化器/调度器/状态工具的框架行为，涉及浮点张量的设备对照以 CPU 结果为 Golden，用 torch.testing.assert_close 校验 NPU 路径一致。
- prepend 语义未覆盖：实测 nn.Module.register_load_state_dict_post_hook（torch 2.11）不提供 prepend 参数（与 Optimizer 版本不同），
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
class TestAveragedModelregister_load_state_dict_post_hook(TestCase):
    """Test cases for torch.optim.swa_utils.AveragedModel.register_load_state_dict_post_hook (NPU/CUDA portable).

    AveragedModel inherits this hook from nn.Module (NOT from Optimizer), so
    the signature differs from the optimizer flavour. Empirical contract
    (torch 2.11, probed on NPU):
      - hook signature is hook(module, incompatible_keys) where
        incompatible_keys is the _IncompatibleKeys(missing_keys,
        unexpected_keys) NamedTuple and the hook may edit it in place;
      - the hook must return None: a non-None return makes load_state_dict()
        raise AssertionError (probed; the state_dict flavour raises
        RuntimeError instead);
      - the hook fires AFTER the payload is applied, so it already observes
        the freshly loaded weights on the module;
      - clearing missing_keys/unexpected_keys inside the hook makes a strict
        load pass that would otherwise fail;
      - no prepend kwarg exists (TypeError), hooks fire in registration order;
      - a non-callable registers fine (no _from_public_api stamping on this
        API) and raises TypeError when fired.
    """

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_module_hook_trigger(self):
        """The registered module hook fires at the documented lifecycle point."""
        am = torch.optim.swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        calls = []
        if 'load_state_dict' == 'state_dict':
            h = am.register_state_dict_post_hook(lambda m, sd: calls.append(1))
            am.state_dict()
        else:
            h = am.register_load_state_dict_post_hook(
                lambda m, incompatible_keys: calls.append(1))
            am.load_state_dict(am.state_dict())
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        h.remove()

    def test_hook_signature_and_loaded_weights_visible(self):
        """Hook gets (this module, _IncompatibleKeys) and sees the loaded weights."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        seen = []
        am.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: seen.append(
                (module, incompatible_keys, module.module.weight.detach().clone())))
        sd = am.state_dict()
        sd["module.weight"] = torch.full((2, 2), 7.0)
        sd["module.bias"] = torch.full((2,), 3.0)
        am.load_state_dict(sd)

        self.assertEqual(len(seen), 1, "load_state_dict() must fire the hook exactly once")
        module, incompatible_keys, weight_seen = seen[0]
        self.assertIs(module, am, "the hook must receive this very AveragedModel")
        # The payload is the documented NamedTuple of key lists.
        self.assertEqual(incompatible_keys._fields, ("missing_keys", "unexpected_keys"))
        self.assertEqual(list(incompatible_keys.missing_keys), [])
        self.assertEqual(list(incompatible_keys.unexpected_keys), [])
        # The hook runs after loading, so the new values are already visible.
        torch.testing.assert_close(weight_seen, torch.full((2, 2), 7.0),
                                   msg="the hook must observe the freshly loaded weights")
        torch.testing.assert_close(am.module.bias.detach(), torch.full((2,), 3.0),
                                   msg="load_state_dict must apply the payload to the module")

    def test_hook_sees_missing_and_unexpected_keys(self):
        """strict=False load reports missing/unexpected keys to the hook."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        seen = []
        am.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: seen.append(
                (list(incompatible_keys.missing_keys), list(incompatible_keys.unexpected_keys))))
        # Payload missing n_averaged/module.bias and carrying an extra key.
        payload = {"module.weight": torch.zeros(2, 2), "extra.unknown": torch.zeros(1)}
        am.load_state_dict(payload, strict=False)
        self.assertEqual(len(seen), 1)
        missing, unexpected = seen[0]
        self.assertCountEqual(missing, ["n_averaged", "module.bias"],
                              "keys absent from the payload must be reported as missing")
        self.assertEqual(unexpected, ["extra.unknown"],
                         "keys unknown to the module must be reported as unexpected")
        # The same payload is rejected by strict=True (no hook interference).
        am2 = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        with self.assertRaises(RuntimeError):
            am2.load_state_dict(payload, strict=True)

    def test_hook_can_clear_incompatible_keys(self):
        """A hook clearing incompatible_keys in place lets a strict load pass."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))

        def clear_keys(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        am.register_load_state_dict_post_hook(clear_keys)
        # This payload would fail strict=True without the hook; with the hook
        # clearing the mismatch report the load must succeed silently.
        am.load_state_dict({"module.weight": torch.zeros(2, 2)}, strict=True)
        torch.testing.assert_close(am.module.weight.detach(), torch.zeros(2, 2),
                                   msg="the load itself must still be applied")

    def test_hook_non_none_return_raises(self):
        """The load path enforces hook(...) -> None with an AssertionError."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        am.register_load_state_dict_post_hook(lambda m, ik: {"replaced": 1})
        with self.assertRaisesRegex(AssertionError, "not.*expected to return"):
            am.load_state_dict(am.state_dict())

    def test_hook_registration_order_and_no_prepend(self):
        """Hooks fire in registration order; the Module API has no prepend kwarg."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        order = []
        am.register_load_state_dict_post_hook(lambda m, ik: order.append("A"))
        am.register_load_state_dict_post_hook(lambda m, ik: order.append("B"))
        am.load_state_dict(am.state_dict())
        self.assertEqual(order, ["A", "B"],
                         "without prepend support the firing order must follow registration")
        # The Optimizer flavour has prepend=..., the nn.Module flavour does not.
        with self.assertRaises(TypeError):
            am.register_load_state_dict_post_hook(lambda m, ik: None, prepend=True)

    def test_hook_removal_stops_firing(self):
        """handle.remove() unregisters the hook; load_state_dict keeps working."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        calls = []
        h = am.register_load_state_dict_post_hook(lambda m, ik: calls.append(1))
        am.load_state_dict(am.state_dict())
        self.assertEqual(len(calls), 1)
        h.remove()
        am.load_state_dict(am.state_dict())
        self.assertEqual(len(calls), 1, "removed hook must not fire again")
        # load_state_dict() itself remains fully functional after removal.
        self.assertEqual(int(am.n_averaged), 0)

    def test_hook_non_callable_typeerror_when_fired(self):
        """A non-callable registers fine and raises TypeError when fired."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        # Unlike register_state_dict_post_hook, this API does not stamp
        # attributes on the hook, so a str registers without error (probed).
        h = am.register_load_state_dict_post_hook("not-callable")
        self.assertIsInstance(h, torch.utils.hooks.RemovableHandle)
        with self.assertRaises(TypeError):
            am.load_state_dict(am.state_dict())
        h.remove()
        # After removing the bad hook, load_state_dict() works again.
        am.load_state_dict(am.state_dict())

    def test_strict_load_still_validates_payload(self):
        """strict=True still rejects bad payloads even with a hook installed."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2))
        am.register_load_state_dict_post_hook(lambda m, ik: None)
        with self.assertRaises(RuntimeError):
            am.load_state_dict({"module.weight": torch.zeros(2, 2), "bogus": torch.zeros(1)},
                               strict=True)

    def test_hook_device_resident_load(self):
        """The hook fires for a device-resident AveragedModel loading device tensors."""
        am = swa_utils.AveragedModel(torch.nn.Linear(2, 2).to(self.device))
        seen = []
        am.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: seen.append(module))
        sd = am.state_dict()
        sd["module.weight"] = torch.full((2, 2), 5.0, device=self.device)
        am.load_state_dict(sd)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], am)
        # The loaded tensor stays on the module's device with the right values.
        self.assertEqual(am.module.weight.device.type, self.device_name)
        torch.testing.assert_close(am.module.weight.detach().cpu(), torch.full((2, 2), 5.0),
                                   msg="device-resident load must land in the module")


if __name__ == "__main__":
    run_tests()
