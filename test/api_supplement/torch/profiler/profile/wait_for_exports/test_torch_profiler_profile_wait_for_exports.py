# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.wait_for_exports 接口功能正确性
API 名称：torch.profiler.profile.wait_for_exports
API 签名：profile.wait_for_exports()

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空输入                         | 未覆盖（见下）                                     |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | 不同输入参数类型                     | 未覆盖（见下）                                     |
| 传参与不传参     | 无参调用与多余参数调用               | 已覆盖（noop_non_cuspy_lifecycle、unexpected_arg_raises） |
| 等价类/边界值    | 多生命周期阶段与重复调用幂等         | 已覆盖（noop_non_cuspy_lifecycle、cuspy_idempotent_second_call） |
| 正常传参场景     | 非 cuspy 空操作及 cuspy join 重置    | 已覆盖（noop_non_cuspy_lifecycle、cuspy_observer_join_and_reset） |
| 异常传参场景     | 传入多余参数触发 TypeError           | 已覆盖（unexpected_arg_raises）                    |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 空/非空：接口无外部参数，不涉及空/非空输入。
- 枚举选项：接口无枚举类型参数。
- 参数类型：接口不接收外部参数，多余参数已在异常传参场景验证。
- 混合设备类型：接口不接收 Tensor 输入，不涉及多设备混合场景。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
from unittest.mock import MagicMock

import torch
from torch.profiler import ProfilerActivity

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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
class TestProfilerProfileWaitForExports(TestCase):
    """Test cases for torch.profiler.profile.wait_for_exports."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_noop_non_cuspy_lifecycle(self):
        """wait_for_exports() is a safe no-op returning None before, during, and after context."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        self.assertIsNone(prof.wait_for_exports())

        x = torch.ones((4, 4), device=self.device)
        with prof:
            _ = x + x
            self.assertIsNone(prof.wait_for_exports())

        self.assertIsNone(prof.wait_for_exports())

    def test_cuspy_observer_join_and_reset(self):
        """With _use_cuspy=True, wait_for_exports joins observer and clears cuspy state."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        observer = MagicMock()
        prof._use_cuspy = True
        prof._cuspy_profiler_observer = observer
        prof._cuspy_window_id = 42

        ret = prof.wait_for_exports()
        self.assertIsNone(ret)
        observer.join.assert_called_once_with()
        self.assertIsNone(prof._cuspy_profiler_observer)
        self.assertIsNone(prof._cuspy_window_id)

    def test_cuspy_idempotent_second_call(self):
        """A second wait_for_exports() call does not invoke observer.join() again."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        observer = MagicMock()
        prof._use_cuspy = True
        prof._cuspy_profiler_observer = observer
        prof._cuspy_window_id = 7

        self.assertIsNone(prof.wait_for_exports())
        self.assertIsNone(prof.wait_for_exports())
        self.assertEqual(observer.join.call_count, 1)
        self.assertIsNone(prof._cuspy_profiler_observer)
        self.assertIsNone(prof._cuspy_window_id)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to wait_for_exports() raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.wait_for_exports(1)


if __name__ == "__main__":
    run_tests()
