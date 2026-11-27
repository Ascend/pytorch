# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.random.thread_safe_generator 接口功能正确性
API 名称：torch.random.thread_safe_generator
API 签名：torch.random.thread_safe_generator() -> torch.Generator | None

覆盖维度表：
| 覆盖维度         | 说明                                                       | 覆盖情况 |
|------------------|------------------------------------------------------------|----------|
| 返回值/身份      | 主进程返回 None；线程 worker 返回其 thread-local 生成器     | 已覆盖 |
| 枚举选项         | worker_method 为 "thread" / "multiprocessing"              | 已覆盖 |
| 等价类/边界值    | 线程 worker 的 rng 为容器 / 为 None                        | 已覆盖 |
| 引用同一性       | 线程 worker 场景返回的正是 worker_info.rng.torch_generator | 已覆盖 |
| 副作用可证伪     | 返回的生成器驱动确定性随机流                               | 已覆盖 |
| 正常传参场景     | 各 worker 上下文返回与文档一致的取值                       | 已覆盖 |
| 异常传参场景     | 该接口无入参、无稳定异常路径                               | 未覆盖（无异常路径） |
| 混合设备类型     | 生成器以默认 CPU 生成器承载，无多设备混用                  | 未覆盖（不适用） |

未覆盖项及原因：
- 异常传参场景：thread_safe_generator 无参数，无稳定异常路径。
- 混合设备类型：该接口只暴露 worker 的 thread-local 生成器（CPU），不涉及异构 Tensor 输入。
- 线程 worker 上下文通过 patch torch.utils.data.get_worker_info 构造（真实线程 worker 上下文
  依赖尚未接入 DataLoader 的线程 worker 模式），断言以「返回同一生成器对象」为准，可证伪。

注意：该接口为全局/上下文状态类框架工具；主进程无 worker_info，返回 None。
     返回的生成器为 CPU 生成器，测试用其驱动 torch.randint 以验证副作用真实生效。
"""
import os
import random
import unittest
import unittest.mock

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
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
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == "npu":
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

from torch.utils.data._utils.worker import WorkerInfo, _RNG


def _thread_worker(generator):
    """Build a thread-worker WorkerInfo exposing the given torch generator."""
    rng = None if generator is None else _RNG(random.Random(0), generator, None)
    return WorkerInfo(id=0, num_workers=1, seed=0, dataset=None, rng=rng, worker_method="thread")


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRandomThreadSafeGeneratorUt(TestCase):
    """Functional tests for torch.random.thread_safe_generator (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_main_process_returns_none(self):
        """Outside any worker, the function returns None."""
        self.assertIsNone(torch.random.thread_safe_generator())

    def test_thread_worker_returns_worker_generator(self):
        """In a thread worker it returns that worker's own torch generator."""
        generator = torch.Generator()
        worker = _thread_worker(generator)
        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=worker):
            self.assertIs(torch.random.thread_safe_generator(), generator)

    def test_thread_worker_without_rng_returns_none(self):
        """A thread worker without an rng container yields None."""
        worker = _thread_worker(None)
        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=worker):
            self.assertIsNone(torch.random.thread_safe_generator())

    def test_multiprocessing_worker_returns_none(self):
        """A multiprocessing worker is not a thread worker, so None is returned."""
        worker = WorkerInfo(
            id=0, num_workers=1, seed=0, dataset=None,
            rng=_RNG(random.Random(0), torch.Generator(), None),
            worker_method="multiprocessing",
        )
        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=worker):
            self.assertIsNone(torch.random.thread_safe_generator())

    def test_distinct_workers_expose_distinct_generators(self):
        """Each worker surfaces its own generator, not a shared default."""
        first, second = torch.Generator(), torch.Generator()
        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=_thread_worker(first)):
            self.assertIs(torch.random.thread_safe_generator(), first)
        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=_thread_worker(second)):
            self.assertIs(torch.random.thread_safe_generator(), second)

    def test_returned_generator_drives_random(self):
        """The returned generator actually drives reproducible random draws."""
        generator = torch.Generator()
        with unittest.mock.patch("torch.utils.data.get_worker_info", return_value=_thread_worker(generator)):
            returned = torch.random.thread_safe_generator()
            returned.manual_seed(2024)
            first = torch.randint(0, 1000, (8,), generator=returned)
            returned.manual_seed(2024)
            second = torch.randint(0, 1000, (8,), generator=returned)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    run_tests()
