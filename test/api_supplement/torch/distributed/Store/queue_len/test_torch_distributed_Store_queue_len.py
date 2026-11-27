# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.queue_len 接口功能正确性
API 名称：torch.distributed.Store.queue_len
API 签名：Store.queue_len(self, key: str) -> int

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 不存在/空队列返回 0，入队后返回精确长度    | 已覆盖（test_missing_push_and_pop_queue_len）          |
| 枚举选项       | TCPStore 与 PrefixStore 两种队列宿主       | 已覆盖（test_missing_push_and_pop_queue_len、          |
|                |                                            | test_multiple_queues_and_prefix_isolation）            |
| 参数类型       | str 键名与非 str 键名异常                  | 已覆盖（test_invalid_key_type_raises）                 |
| 传参与不传参   | key 为必选参数，验证缺参抛 TypeError       | 已覆盖（test_invalid_key_type_raises）                 |
| 等价类/边界值  | 0→1→N→0 全生命周期、多队列独立计数         | 已覆盖（test_missing_push_and_pop_queue_len、          |
|                |                                            | test_multiple_queues_and_prefix_isolation）            |
| 正常传参场景   | 入队/出队动态校验 queue_len 精确整数值     | 已覆盖（全部正常用例）                                 |
| 异常传参场景   | 非法键类型、缺参、多余参数触发 TypeError   | 已覆盖（test_invalid_key_type_raises）                 |
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 队列长度查询不接收 Tensor 参数）         |

未覆盖项及原因：
- 混合设备类型：Store.queue_len 仅查询分布式队列元素个数，不接收 Tensor 参数。

注意：本接口属于分布式 Store 扩展 API，在单进程 TCPStore（libuv）上验证，无浮点 Tensor 输出；
     通过入队、出队前后的精确整数长度断言验证功能正确性。
"""
import os
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
class TestStoreQueueLen(TestCase):
    def _make_tcp_store(self):
        return dist.TCPStore(
            "127.0.0.1", 0, 1, is_master=True, wait_for_workers=False, use_libuv=True
        )

    def test_missing_push_and_pop_queue_len(self):
        store = self._make_tcp_store()
        missing_len = store.queue_len("missing_queue")
        self.assertIsInstance(missing_len, int)
        self.assertEqual(missing_len, 0)

        key = "jobs"
        for idx in range(1, 5):
            store.queue_push(key, f"item_{idx}")
            self.assertEqual(store.queue_len(key), idx)

        for remaining in range(3, -1, -1):
            store.queue_pop(key, block=False)
            self.assertEqual(store.queue_len(key), remaining)

    def test_multiple_queues_and_prefix_isolation(self):
        base_store = self._make_tcp_store()
        base_store.queue_push("q_a", "a1")
        base_store.queue_push("q_a", "a2")
        base_store.queue_push("q_b", "b1")
        self.assertEqual(base_store.queue_len("q_a"), 2)
        self.assertEqual(base_store.queue_len("q_b"), 1)
        self.assertEqual(base_store.queue_len("q_c"), 0)

        p1, p2 = dist.PrefixStore("ns1", base_store), dist.PrefixStore("ns2", base_store)
        p1.queue_push("q", "x")
        p1.queue_push("q", "y")
        p2.queue_push("q", "z")
        self.assertEqual(p1.queue_len("q"), 2)
        self.assertEqual(p2.queue_len("q"), 1)

    def test_invalid_key_type_raises(self):
        store = self._make_tcp_store()
        for args in [(123,), (), ("q", True)]:
            with self.assertRaises(TypeError):
                store.queue_len(*args)


if __name__ == "__main__":
    run_tests()
