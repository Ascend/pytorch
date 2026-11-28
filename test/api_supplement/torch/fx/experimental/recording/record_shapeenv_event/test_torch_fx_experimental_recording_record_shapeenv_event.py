# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.recording.record_shapeenv_event 接口功能正确性
API 名称：torch.fx.experimental.recording.record_shapeenv_event
API 签名：record_shapeenv_event(*, save_tracked_fakes=False, name=None) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：关闭记录和异常事件回滚 |
| 正常传参场景 | 已覆盖：真实 ShapeEnv 方法记录、返回值及 tracked_fakes 快照 |
| 异常传参场景 | 已覆盖：原异常传播、记录标记与日志嵌套恢复 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

from torch.fx.experimental import recording
from torch.fx.experimental.symbolic_shapes import ShapeEnv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_record_shapeenv_event_recorded_method_returns_value_and_saves_snapshot(self):
        env = ShapeEnv(should_record_events=True, tracked_fakes=[])
        self.addCleanup(env.cleanup)

        @recording.record_shapeenv_event(save_tracked_fakes=True, name="ut_event")
        def update(self, value, *, step=2):
            self.ut_value = value + step
            return self.ut_value

        before = len(env.events)
        self.assertEqual(update(env, 3, step=4), 7)
        self.assertEqual(env.ut_value, 7)
        self.assertEqual(len(env.events), before + 1)
        event = env.events[-1]
        self.assertEqual(event.name, "ut_event")
        self.assertEqual(event.tracked_fakes, [])
        self.assertFalse(env.is_recording)

    def test_record_shapeenv_event_disabled_recording_and_error_rollback_preserve_state(self):
        env = ShapeEnv(should_record_events=False, tracked_fakes=[])
        self.addCleanup(env.cleanup)

        @recording.record_shapeenv_event()
        def identity(self, value):
            return value + 1

        before = len(env.events)
        self.assertEqual(identity(env, 5), 6)
        self.assertEqual(len(env.events), before)
        env.should_record_events = True
        nesting = recording.NEST

        @recording.record_shapeenv_event()
        def failure(self):
            raise ValueError("controlled record error")

        with self.assertRaisesRegex(ValueError, "controlled record error"):
            failure(env)
        self.assertEqual(len(env.events), before)
        self.assertFalse(env.is_recording)
        self.assertEqual(recording.NEST, nesting)


if __name__ == "__main__":
    run_tests()
