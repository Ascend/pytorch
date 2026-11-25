# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler.emit_nvtx 接口功能正确性
API 名称：torch.autograd.profiler.emit_nvtx
API 签名：torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 不涉及容器或空 Tensor 输入 | 未覆盖（上下文管理器仅接受布尔标志参数） |
| 枚举选项 | enabled / record_shapes 组合 | 已覆盖（default_attributes_and_invalid_kwarg、disabled_noop、enabled_mocked_lifecycle） |
| 参数类型 | 布尔参数与非法关键字参数 | 已覆盖（default_attributes_and_invalid_kwarg、disabled_noop） |
| 传参与不传参 | 默认与显式传参 | 已覆盖（default_attributes_and_invalid_kwarg、disabled_noop、enabled_mocked_lifecycle） |
| 等价类/边界值 | 禁用状态下重入无操作 | 已覆盖（disabled_noop） |
| 正常传参场景 | 禁用空操作、模拟启停 | 已覆盖（disabled_noop、enabled_mocked_lifecycle） |
| 异常传参场景 | 未知关键字参数 | 已覆盖（default_attributes_and_invalid_kwarg） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（上下文管理器不接收 Tensor 参数） |

未覆盖项及原因：
- 空/非空：接口仅接收 enabled 与 record_shapes 布尔标志，无容器或 Tensor 输入。
- 混合设备类型：接口不接收 Tensor 参数，不涉及跨设备 Tensor 混合输入。
- 真实设备实测：需要 CUDA 环境验证 NVTX 实际采集，当前用例仅覆盖 mock 生命周期。

注意：本接口为 NVTX 上下文管理器，无浮点 Tensor 输出，
     按 API 语义使用属性状态、钩子调用副作用与异常断言验证。
"""
import os
import unittest
from unittest.mock import patch

import torch
from torch.autograd.profiler import emit_nvtx
from torch._C._profiler import ProfilerState

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEmitNvtx(TestCase):
    """Test cases for torch.autograd.profiler.emit_nvtx."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_default_attributes_and_invalid_kwarg(self):
        """Default flags are initialized properly and unknown kwargs raise TypeError."""
        cm = emit_nvtx()
        self.assertEqual((cm.enabled, cm.record_shapes, cm.entered), (True, False, False))
        with self.assertRaises(TypeError):
            emit_nvtx(unknown_flag=True)

    def test_disabled_noop(self):
        """Disabled emit_nvtx is a no-op and allows nested entry without enabling profiler."""
        cm = emit_nvtx(enabled=False, record_shapes=True)
        self.assertFalse(cm.enabled)
        self.assertTrue(cm.record_shapes)
        with patch("torch.cuda.synchronize") as m_sync, patch("torch.autograd.profiler._enable_profiler") as m_en:
            with cm as entered_val:
                self.assertIsNone(entered_val)
                with cm:
                    self.assertFalse(cm.entered)
            m_sync.assert_not_called()
            m_en.assert_not_called()

    def test_enabled_mocked_lifecycle(self):
        """Enabled emit_nvtx triggers sync, start/stop hooks, and NVTX profiler config."""
        captured = []
        cm = emit_nvtx(enabled=True, record_shapes=True)
        with patch("torch.cuda.synchronize") as m_sync, \
                patch("torch.autograd.profiler._run_on_profiler_start") as m_start, \
                patch("torch.autograd.profiler._run_on_profiler_stop") as m_stop, \
                patch("torch.autograd.profiler._enable_profiler") as m_en, \
                patch("torch.autograd.profiler._disable_profiler") as m_dis, \
                patch("torch.autograd.profiler.ProfilerConfig", side_effect=lambda *a: captured.append(a)):
            with cm as entered_val:
                self.assertIs(entered_val, cm)
                self.assertTrue(cm.entered)
            self.assertEqual(m_sync.call_count, 2)
            for mock_fn in (m_start, m_en, m_dis, m_stop):
                mock_fn.assert_called_once()
            self.assertEqual(captured[0][0], ProfilerState.NVTX)
            self.assertIs(captured[0][1], True)


if __name__ == "__main__":
    run_tests()
