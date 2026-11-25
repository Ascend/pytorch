# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profiler.schedule 接口功能正确性
API 名称：torch.profiler.profiler.schedule
API 签名：torch.profiler.profiler.schedule(*, wait, warmup, active, repeat=0, skip_first=0, skip_first_wait=0)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空容器输入                     | 未覆盖（见下）                                     |
| 枚举选项         | 返回值 ProfilerAction 枚举覆盖       | 已覆盖（infinite_and_repeat_cycles、skip_first_options） |
| 参数类型         | 整数关键字参数与非法位置参数         | 已覆盖（infinite_and_repeat_cycles、invalid_args_and_negative_step_raise） |
| 传参与不传参     | 可选参数默认与显式传参               | 已覆盖（zero_warmup_warns、skip_first_options）    |
| 等价类/边界值    | repeat=0/1、warmup=0 告警、负步数    | 已覆盖（infinite_and_repeat_cycles、zero_warmup_warns） |
| 正常传参场景     | 无限/有限周期及 skip_first 调度      | 已覆盖（infinite_and_repeat_cycles、skip_first_options 等） |
| 异常传参场景     | 非法参数值、负步数及位置传参报错     | 已覆盖（invalid_args_and_negative_step_raise）     |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 空/非空：接口仅接收整数调度参数，无容器或空值输入维度。
- 混合设备类型：纯 Python 步数调度函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
import warnings

import torch
from torch.profiler.profiler import ProfilerAction, schedule

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
class TestProfilerSchedule(TestCase):
    """Test cases for torch.profiler.profiler.schedule."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_infinite_and_repeat_cycles(self):
        """Schedule cycles repeat infinitely when repeat=0 and stop after repeat=1."""
        n, w, r, s = (
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
        )
        sched_inf = schedule(wait=1, warmup=1, active=2, repeat=0)
        self.assertEqual([sched_inf(i) for i in range(8)], [n, w, r, s, n, w, r, s])

        sched_once = schedule(wait=1, warmup=1, active=2, repeat=1)
        self.assertEqual([sched_once(i) for i in range(8)], [n, w, r, s, n, n, n, n])

    def test_skip_first_options(self):
        """skip_first and skip_first_wait offset initial schedule actions as expected."""
        n, w, s = ProfilerAction.NONE, ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE
        sched_skip = schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=2)
        self.assertEqual([sched_skip(i) for i in range(7)], [n, n, n, w, s, n, n])

        sched_skip_wait = schedule(
            wait=2, warmup=1, active=1, repeat=0, skip_first=1, skip_first_wait=1
        )
        self.assertEqual([sched_skip_wait(i) for i in range(7)], [n, w, s, n, n, w, s])

    def test_zero_warmup_warns(self):
        """Setting warmup=0 emits a UserWarning and transitions directly to RECORD."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sched = schedule(wait=1, warmup=0, active=2)

        user_warnings = [item for item in caught if issubclass(item.category, UserWarning)]
        self.assertGreaterEqual(len(user_warnings), 1)
        self.assertEqual(
            [sched(i) for i in range(3)],
            [ProfilerAction.NONE, ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE],
        )

    def test_invalid_args_and_negative_step_raise(self):
        """Invalid schedule bounds, negative steps, and positional args raise errors."""
        sched = schedule(wait=1, warmup=1, active=1)
        with self.assertRaises(AssertionError):
            sched(-1)

        invalid_kwargs = [
            {"wait": -1, "warmup": 1, "active": 1},
            {"wait": 1, "warmup": -1, "active": 1},
            {"wait": 1, "warmup": 1, "active": 0},
            {"wait": 1, "warmup": 1, "active": 1, "repeat": -1},
            {"wait": 1, "warmup": 1, "active": 1, "skip_first": -1},
        ]
        for kwargs in invalid_kwargs:
            with self.assertRaises(AssertionError):
                schedule(**kwargs)

        with self.assertRaises(TypeError):
            schedule(1, 1, 2)


if __name__ == "__main__":
    run_tests()
