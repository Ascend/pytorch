# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.barrier 同步屏障接口功能正确性
API 名称：torch.distributed.Store.barrier
API 签名：Store.barrier(self, key: str, world_size: int, timeout: timedelta | None = None) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 非空 key 与不同 barrier key 独立同步 | 已覆盖（test_single_worker_barrier、test_multithread_barrier_synchronization） |
| 枚举选项 | 无枚举选项参数 | 未覆盖（Store.barrier 无枚举入参） |
| 参数类型 | str key、int world_size、timedelta/None timeout | 已覆盖（test_single_worker_barrier、test_timeout_and_invalid_args） |
| 传参与不传参 | 省略 timeout（默认超时）与显式传入 timedelta | 已覆盖（test_single_worker_barrier、test_timeout_and_invalid_args） |
| 等价类/边界值 | world_size=1/3、超时 | 已覆盖（test_single_worker_barrier、test_multithread_barrier_synchronization） |
| 正常传参场景 | 多线程到达 barrier 后才继续执行，验证真实同步副作用 | 已覆盖（test_multithread_barrier_synchronization） |
| 异常传参场景 | 参与者不足超时抛 RuntimeError；非法参数类型抛 TypeError | 已覆盖（test_timeout_and_invalid_args） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（控制面存储同步屏障，无 Tensor 参数） |

未覆盖项及原因：
- 枚举选项：Store.barrier 仅接收 key、world_size、timeout，无枚举参数。
- 混合设备类型：接口为控制面同步原语，不接收 Tensor 参数。
- 多卡 HCCL 通信：依据分布式规范，Store 系单进程（多线程）测试，不依赖 HCCL 后端。

注意：多线程用例通过线程到达计数与释放顺序断言 barrier 真实阻塞并同步，而非仅断言不报错；
     超时用例显式传入毫秒级短超时以确保快速失败。
"""
from datetime import timedelta
import os
import threading
import time
import unittest

import torch
import torch.distributed as dist

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        DEVICE_NAME = "npu" if torch.npu.is_available() else None
    except (ImportError, AttributeError):
        DEVICE_NAME = None
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    from unittest import TestCase
    run_tests = unittest.main


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestStoreBarrier(TestCase):
    def test_single_worker_barrier(self):
        store = dist.HashStore()
        self.assertIsNone(store.barrier("b_single_default", 1))
        self.assertIsNone(store.barrier("b_single_timeout", 1, timedelta(seconds=5)))

    def test_multithread_barrier_synchronization(self):
        store = dist.HashStore()
        world_size = 3
        arrived, passed, errors = [], [], []
        lock = threading.Lock()
        release_last = threading.Event()

        def worker(idx):
            try:
                if idx == world_size - 1:
                    # Delay final participant so earlier threads must block at the barrier.
                    release_last.wait(timeout=5.0)
                with lock:
                    arrived.append(idx)
                store.barrier("b_multi", world_size, timedelta(seconds=10))
                with lock:
                    passed.append(idx)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(world_size)]
        for t in threads:
            t.start()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with lock:
                if len(arrived) == world_size - 1:
                    break
            time.sleep(0.01)

        # Early threads must remain blocked until the final participant arrives.
        time.sleep(0.05)
        with lock:
            self.assertEqual(len(arrived), world_size - 1)
            self.assertEqual(len(passed), 0)

        release_last.set()
        for t in threads:
            t.join(timeout=10.0)
            self.assertFalse(t.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(sorted(passed), list(range(world_size)))

    def test_timeout_and_invalid_args(self):
        store = dist.HashStore()
        with self.assertRaises(RuntimeError):
            store.barrier("b_timeout", 2, timedelta(milliseconds=50))
        for args in [("only_key",), (123, 1), ("b_bad_ws", "two")]:
            with self.assertRaises(TypeError):
                store.barrier(*args)


if __name__ == "__main__":
    run_tests()
