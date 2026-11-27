# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.queue_pop 接口功能正确性
API 名称：torch.distributed.Store.queue_pop
API 签名：Store.queue_pop(self, key: str, block: bool = True) -> bytes

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 非空队列出队、空队列非阻塞抛错、空字节载荷 | 已覆盖（test_fifo_binary_and_prefix_isolation、        |
|                |                                            | test_non_blocking_empty_and_invalid_args_raises）      |
| 枚举选项       | block=True（默认与显式）/ block=False      | 已覆盖（test_fifo_binary_and_prefix_isolation、        |
|                |                                            | test_blocking_pop_unblocks_on_push、                   |
|                |                                            | test_non_blocking_empty_and_invalid_args_raises）      |
| 参数类型       | str 键名、bool 标志及非法类型异常          | 已覆盖（test_non_blocking_empty_and_invalid_args_raises）|
| 传参与不传参   | 省略 block（默认 True）与显式传入 block    | 已覆盖（test_fifo_binary_and_prefix_isolation、        |
|                |                                            | test_blocking_pop_unblocks_on_push）                   |
| 等价类/边界值  | FIFO 顺序、空字节、二进制零字节、PrefixStore | 已覆盖（test_fifo_binary_and_prefix_isolation）       |
| 正常传参场景   | 同步/非阻塞出队与并发线程延迟入队唤醒      | 已覆盖（test_fifo_binary_and_prefix_isolation、        |
|                |                                            | test_blocking_pop_unblocks_on_push）                   |
| 异常传参场景   | 空队列 block=False 抛 RuntimeError、错参   | 已覆盖（test_non_blocking_empty_and_invalid_args_raises）|
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 队列出队不接收 Tensor 参数）             |

未覆盖项及原因：
- 混合设备类型：Store.queue_pop 仅从分布式队列弹出字节数据，不接收 Tensor 参数。

注意：本接口属于分布式 Store 扩展 API，在单进程 TCPStore（libuv）上验证，无浮点 Tensor 输出；
     通过 FIFO 精确字节值、队列长度递减、阻塞唤醒与异常断言验证功能正确性。
"""
import datetime
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
class TestStoreQueuePop(TestCase):
    def _make_tcp_store(self, timeout_seconds=10):
        return dist.TCPStore(
            "127.0.0.1",
            0,
            1,
            is_master=True,
            timeout=datetime.timedelta(seconds=timeout_seconds),
            wait_for_workers=False,
            use_libuv=True,
        )

    def test_fifo_binary_and_prefix_isolation(self):
        store = self._make_tcp_store()
        key = "fifo_q"
        payloads = [b"first", b"", b"\x00\x01\xfe\xff", "last_str".encode("utf-8")]
        for item in payloads:
            store.queue_push(key, item)

        self.assertEqual(store.queue_len(key), len(payloads))
        first = store.queue_pop(key)
        second = store.queue_pop(key, block=False)
        third = store.queue_pop(key, block=True)
        fourth = store.queue_pop(key, False)

        self.assertIsInstance(first, bytes)
        self.assertEqual([first, second, third, fourth], payloads)
        self.assertEqual(store.queue_len(key), 0)

        p1, p2 = dist.PrefixStore("p1", store), dist.PrefixStore("p2", store)
        p1.queue_push("q", "from_p1")
        p2.queue_push("q", "from_p2")
        self.assertEqual(p1.queue_pop("q", block=False), b"from_p1")
        self.assertEqual(p2.queue_len("q"), 1)
        self.assertEqual(p2.queue_pop("q", block=False), b"from_p2")

    def test_blocking_pop_unblocks_on_push(self):
        store = self._make_tcp_store(timeout_seconds=10)
        producer_store = store.clone()
        key = "delayed_q"

        def _delayed_push():
            time.sleep(0.05)
            producer_store.queue_push(key, b"delayed_value")

        worker = threading.Thread(target=_delayed_push, daemon=True)
        worker.start()
        result = store.queue_pop(key, block=True)
        worker.join(timeout=5.0)

        self.assertEqual(result, b"delayed_value")
        self.assertEqual(store.queue_len(key), 0)

    def test_non_blocking_empty_and_invalid_args_raises(self):
        store = self._make_tcp_store()
        with self.assertRaises(RuntimeError):
            store.queue_pop("empty_missing_key", block=False)

        store.queue_push("drain_q", "only_item")
        self.assertEqual(store.queue_pop("drain_q", block=False), b"only_item")
        with self.assertRaises(RuntimeError):
            store.queue_pop("drain_q", block=False)

        for args in [(123,), (), ("q", "not_a_bool")]:
            with self.assertRaises(TypeError):
                store.queue_pop(*args)


if __name__ == "__main__":
    run_tests()
