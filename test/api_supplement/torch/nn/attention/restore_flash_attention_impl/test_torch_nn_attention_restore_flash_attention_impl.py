# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.attention.restore_flash_attention_impl 接口功能正确性
API 名称：torch.nn.attention.restore_flash_attention_impl
API 签名：restore_flash_attention_impl(_raise_warn: bool = True) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无自定义实现处于激活态与有自定义实现处于激活态两分支 | 已覆盖（restore_without_active_impl_warns_and_keeps_default/
restore_after_activation_removes_custom_impl） |
| 枚举选项 | _raise_warn=True/False：默认恢复告警与静默恢复 | 已覆盖（restore_without_active_impl_warns_and_keeps_default/
restore_silent_mode_no_warning） |
| 参数类型 | 返回 None；内部句柄 remove() 副作用经自定义句柄记录 | 已覆盖（restore_without_active_impl_warns_and_keeps_default/
restore_after_activation_removes_custom_impl） |
| 传参与不传参 | 默认 _raise_warn=True 与显式 _raise_warn=False | 已覆盖（restore_without_active_impl_warns_and_keeps_default/
restore_silent_mode_no_warning） |
| 等价类/边界值 | 重复恢复第二次再次告警；activate 切换实现时隐式恢复旧句柄；register_fn 返回 None 句柄 | 已覆盖（restore_twice_warns_on_second_call/
restore_is_called_when_activating_another_impl/
restore_with_none_handle_still_resets） |
| 正常传参场景 | 恢复后 current_flash_attention_impl() 返回 None（默认 FA2 态） | 已覆盖（restore_after_activation_removes_custom_impl） |
| 异常传参场景 | 本接口无抛错契约（告警经 logging 输出），非告警分支静默 | 已覆盖（restore_silent_mode_no_warning） |
| 混合设备类型 | 纯注册表状态 API，不接收 Tensor，无设备输入路径 | 未覆盖（无设备输入路径，原因见文件头说明） |

未覆盖项及原因：
- 混合设备类型：restore_flash_attention_impl 只维护 flash attention 实现注册表状态，不接收
Tensor 输入；副作用通过注册 fake 实现的句柄计数验证，不依赖设备执行。
- 设备无关运行：本类不探测也不初始化任何设备（覆盖基类 setUpClass 的 torch.npu.set_device
选卡），仅在注册表状态上验证契约；仅装 torch_npu 而无加速卡的机器上同样真实运行。
- 内置 FA3/FA4 实现的激活依赖对应 flash 后端包与 CUDA 平台，不在本接口用例范围内，仅以
自定义 fake 实现覆盖注册-激活-恢复生命周期。
"""
import logging
import os
import unittest

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


from torch.nn.attention import (  # noqa: E402
    activate_flash_attention_impl,
    current_flash_attention_impl,
    register_flash_attention_impl,
    restore_flash_attention_impl,
)
from torch.nn.attention import _registry  # noqa: E402


REGISTRY_LOGGER = 'torch.nn.attention._registry'
FAKE_IMPL_NAME = 'SupplementTestFakeImpl'


class _RecordCollector(logging.Handler):
    """Collect log records emitted by the registry logger."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _FakeHandle:
    """Handle that counts remove() invocations for side-effect assertions."""

    def __init__(self):
        self.remove_calls = 0

    def remove(self):
        self.remove_calls += 1


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestAttentionRestoreFlashAttentionImpl(TestCase):
    """Test cases for torch.nn.attention.restore_flash_attention_impl (device independent)."""

    @classmethod
    def setUpClass(cls):
        # Device-independent registry tests must not initialize any device.
        # The torch_npu TestCase base setUpClass calls torch.npu.set_device,
        # which errors on machines where torch_npu is installed but no
        # accelerator exists, so the base device selection is replaced here
        # instead of probing for a device this file never uses. The class
        # then truly runs on CPU-only hosts too, with or without torch_npu.
        cls.npu_device = None

    def setUp(self):
        super().setUp()
        # Restore default state and clean any leftover fake registration.
        restore_flash_attention_impl(_raise_warn=False)
        _registry._FLASH_ATTENTION_IMPLS.pop(FAKE_IMPL_NAME, None)
        self._saved_impls = dict(_registry._FLASH_ATTENTION_IMPLS)
        self._handler = _RecordCollector()
        self._logger = logging.getLogger(REGISTRY_LOGGER)
        self._logger.addHandler(self._handler)

    def tearDown(self):
        restore_flash_attention_impl(_raise_warn=False)
        self._logger.removeHandler(self._handler)
        _registry._FLASH_ATTENTION_IMPLS.clear()
        _registry._FLASH_ATTENTION_IMPLS.update(self._saved_impls)
        super().tearDown()

    def _register_fake_impl(self, handle):
        """Register a fake impl whose activation installs the given handle."""

        def register_fn():
            return handle

        register_flash_attention_impl(FAKE_IMPL_NAME, register_fn=register_fn)
        return register_fn

    # ------------------------------------------------------------------
    # Restore with no custom impl active
    # ------------------------------------------------------------------

    def test_restore_without_active_impl_warns_and_keeps_default(self):
        """Restoring with nothing active logs a warning and keeps default state."""
        self.assertIsNone(current_flash_attention_impl())
        result = restore_flash_attention_impl()
        self.assertIsNone(result)
        self.assertIsNone(current_flash_attention_impl())
        self.assertEqual(len(self._handler.records), 1)
        self.assertIn('no custom impl was activated',
                      self._handler.records[0].getMessage())

    def test_restore_silent_mode_no_warning(self):
        """_raise_warn=False restores silently without logging."""
        result = restore_flash_attention_impl(_raise_warn=False)
        self.assertIsNone(result)
        self.assertEqual(len(self._handler.records), 0)
        self.assertIsNone(current_flash_attention_impl())

    def test_restore_twice_warns_on_second_call(self):
        """A second restore with still nothing active warns again."""
        restore_flash_attention_impl()
        self.assertEqual(len(self._handler.records), 1)
        restore_flash_attention_impl()
        self.assertEqual(len(self._handler.records), 2)
        self.assertIsNone(current_flash_attention_impl())

    # ------------------------------------------------------------------
    # Restore with a custom impl active
    # ------------------------------------------------------------------

    def test_restore_after_activation_removes_custom_impl(self):
        """Restoring after activation removes the custom impl exactly once."""
        handle = _FakeHandle()
        self._register_fake_impl(handle)
        activate_flash_attention_impl(FAKE_IMPL_NAME)
        self.assertEqual(current_flash_attention_impl(), FAKE_IMPL_NAME)
        self.assertEqual(self._handler.records, [])
        restore_flash_attention_impl()
        self.assertEqual(handle.remove_calls, 1)
        self.assertIsNone(current_flash_attention_impl())
        self.assertEqual(self._handler.records, [])

    def test_restore_is_called_when_activating_another_impl(self):
        """Activating a second impl implicitly restores the first handle."""
        first_handle = _FakeHandle()
        second_handle = _FakeHandle()
        self._register_fake_impl(first_handle)
        activate_flash_attention_impl(FAKE_IMPL_NAME)
        self._register_fake_impl(second_handle)
        activate_flash_attention_impl(FAKE_IMPL_NAME)
        self.assertEqual(current_flash_attention_impl(), FAKE_IMPL_NAME)
        self.assertEqual(first_handle.remove_calls, 1)
        restore_flash_attention_impl()
        self.assertEqual(second_handle.remove_calls, 1)
        self.assertEqual(first_handle.remove_calls, 1)
        self.assertIsNone(current_flash_attention_impl())

    def test_restore_with_none_handle_still_resets(self):
        """A register_fn returning None activates nothing; restore warns and resets."""
        register_calls = []

        def register_fn():
            register_calls.append(True)
            return None

        register_flash_attention_impl(FAKE_IMPL_NAME, register_fn=register_fn)
        activate_flash_attention_impl(FAKE_IMPL_NAME)
        # Activation ran register_fn exactly once. A None handle activates
        # nothing, so the default impl stays current; a regression that stored
        # a None handle would surface here as the fake impl name (or as an
        # AttributeError from remove() on None during restore). The registered
        # impl must also stay registered: restore only resets the active one.
        self.assertEqual(len(register_calls), 1)
        self.assertIsNone(current_flash_attention_impl())
        self.assertIn(FAKE_IMPL_NAME, _registry.list_flash_attention_impls())
        restore_flash_attention_impl()
        # Restore must not re-run register_fn and leaves the default state.
        self.assertEqual(len(register_calls), 1)
        self.assertIsNone(current_flash_attention_impl())
        self.assertEqual(len(self._handler.records), 1)
        self.assertIn('no custom impl was activated',
                      self._handler.records[0].getMessage())


if __name__ == "__main__":
    run_tests()
