# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler.KinetoStepTracker.init_step_count 接口功能正确性
API 名称：torch.autograd.profiler.KinetoStepTracker.init_step_count
API 签名：KinetoStepTracker.init_step_count(cls, requester: str) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空字符串与非空 requester | 已覆盖（empty_requester、basic_init） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | 合法字符串与不可哈希类型 | 已覆盖（basic_init、unhashable_requester_raises） |
| 传参与不传参 | 必选 requester 传入与缺参 | 已覆盖（basic_init、missing_arg_raises） |
| 等价类/边界值 | 初始步数 0、递进后初始化、多方与重置 | 已覆盖（after_increment_tracks_current、multiple_requesters、reinit_resets_to_current） |
| 正常传参场景 | 注册单个/多个 requester 并同步步数 | 已覆盖（basic_init、multiple_requesters、reinit_resets_to_current） |
| 异常传参场景 | 缺参、不可哈希对象触发 TypeError | 已覆盖（missing_arg_raises、unhashable_requester_raises） |
| 混合设备类型 | 不涉及 Tensor 设备输入 | 未覆盖（纯 Python 计步器接口，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收字符串 requester，无枚举参数。
- 混合设备类型：接口维护纯 Python 步数状态，不接收 Tensor 参数。

注意：本接口为纯 Python 计步器管理接口，无浮点 Tensor 输出，
     按 API 语义使用状态字典、返回值与异常断言验证。
"""
from collections import defaultdict
import os
import unittest
from unittest.mock import patch

import torch
from torch.autograd.profiler import KinetoStepTracker

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
class TestKinetoStepTrackerInitStepCount(TestCase):
    """Test cases for torch.autograd.profiler.KinetoStepTracker.init_step_count."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved_current_step = KinetoStepTracker._current_step
        self._saved_step_dict = KinetoStepTracker._step_dict.copy()
        KinetoStepTracker._current_step = 0
        KinetoStepTracker._step_dict = defaultdict(int)

    def tearDown(self):
        KinetoStepTracker._current_step = self._saved_current_step
        KinetoStepTracker._step_dict = defaultdict(int, self._saved_step_dict)
        super().tearDown()

    def test_basic_init(self):
        """Initializing a requester records current step 0 and returns None."""
        self.assertIsNone(KinetoStepTracker.init_step_count("ProfilerStep"))
        self.assertIn("ProfilerStep", KinetoStepTracker._step_dict)
        self.assertEqual(KinetoStepTracker._step_dict["ProfilerStep"], 0)
        self.assertEqual(KinetoStepTracker.current_step(), 0)

    def test_empty_requester(self):
        """Empty string requester is accepted and initialized to current step."""
        KinetoStepTracker.init_step_count("")
        self.assertIn("", KinetoStepTracker._step_dict)
        self.assertEqual(KinetoStepTracker._step_dict[""], 0)

    def test_after_increment_tracks_current(self):
        """New requester initialized after increments inherits the global current step."""
        with patch("torch.autograd.profiler._kineto_step"):
            KinetoStepTracker.init_step_count("Opt1")
            KinetoStepTracker.increment_step("Opt1")
            KinetoStepTracker.increment_step("Opt1")
            self.assertEqual(KinetoStepTracker.current_step(), 2)
            KinetoStepTracker.init_step_count("Opt2")
            self.assertEqual(KinetoStepTracker._step_dict["Opt2"], 2)
            self.assertEqual(KinetoStepTracker.increment_step("Opt2"), 3)

    def test_multiple_requesters(self):
        """Multiple distinct requesters are tracked independently in _step_dict."""
        KinetoStepTracker.init_step_count("ReqA")
        KinetoStepTracker.init_step_count("ReqB")
        self.assertEqual(KinetoStepTracker._step_dict, {"ReqA": 0, "ReqB": 0})

    def test_reinit_resets_to_current(self):
        """Re-initializing a lagging requester updates its count to the global step."""
        with patch("torch.autograd.profiler._kineto_step"):
            KinetoStepTracker.init_step_count("Fast")
            KinetoStepTracker.init_step_count("Slow")
            KinetoStepTracker.increment_step("Fast")
            KinetoStepTracker.increment_step("Fast")
            self.assertEqual(KinetoStepTracker._step_dict["Slow"], 0)
            KinetoStepTracker.init_step_count("Slow")
            self.assertEqual(KinetoStepTracker._step_dict["Slow"], 2)

    def test_missing_arg_raises(self):
        """Calling init_step_count without requester raises TypeError."""
        with self.assertRaises(TypeError):
            KinetoStepTracker.init_step_count()

    def test_unhashable_requester_raises(self):
        """Passing an unhashable requester raises TypeError without mutating state."""
        with self.assertRaises(TypeError):
            KinetoStepTracker.init_step_count(["bad"])
        self.assertEqual(len(KinetoStepTracker._step_dict), 0)


if __name__ == "__main__":
    run_tests()
