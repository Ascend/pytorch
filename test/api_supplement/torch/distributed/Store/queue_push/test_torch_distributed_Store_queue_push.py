# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.queue_push 接口功能正确性
API 名称：torch.distributed.Store.queue_push
API 签名：Store.queue_push(self, key: str, value: bytes | str) -> None

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 空字符串/空字节入队与非空多元素入队        | 已覆盖（test_push_str_bytes_and_empty_values）         |
| 枚举选项       | TCPStore 与 PrefixStore 两种队列宿主       | 已覆盖（test_push_str_bytes_and_empty_values、         |
|                |                                            | test_duplicate_values_and_prefix_isolation）           |
| 参数类型       | str 与 bytes 两种合法值类型及非法类型异常  | 已覆盖（test_push_str_bytes_and_empty_values、         |
|                |                                            | test_invalid_args_raises）                             |
| 传参与不传参   | key 与 value 均为必选参数，缺参抛错        | 已覆盖（test_invalid_args_raises）                     |
| 等价类/边界值  | 重复值保留、二进制零字节、多队列隔离       | 已覆盖（test_push_str_bytes_and_empty_values、         |
|                |                                            | test_duplicate_values_and_prefix_isolation）           |
| 正常传参场景   | 入队后核对返回值 None、queue_len 与出队值  | 已覆盖（全部正常用例）                                 |
| 异常传参场景   | 非法键/值类型、缺参触发 TypeError          | 已覆盖（test_invalid_args_raises）                     |
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 队列入队不接收 Tensor 参数）             |

未覆盖项及原因：
- 混合设备类型：Store.queue_push 仅向分布式队列推入字符串/字节数据，不接收 Tensor 参数。

注意：本接口属于分布式 Store 扩展 API，在单进程 TCPStore（libuv）上验证，无浮点 Tensor 输出；
     入队后通过 queue_len 与 queue_pop 精确比对长度与字节内容验证功能正确性。
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
class TestStoreQueuePush(TestCase):
    def _make_tcp_store(self):
        return dist.TCPStore(
            "127.0.0.1", 0, 1, is_master=True, wait_for_workers=False, use_libuv=True
        )

    def test_push_str_bytes_and_empty_values(self):
        store = self._make_tcp_store()
        str_vals = ["hello", "", "unicode_\u6d4b\u8bd5"]
        for val in str_vals:
            self.assertIsNone(store.queue_push("str_q", val))
        self.assertEqual(store.queue_len("str_q"), len(str_vals))
        self.assertEqual(
            [store.queue_pop("str_q", block=False) for _ in str_vals],
            [v.encode("utf-8") for v in str_vals],
        )

        bin_vals = [b"", b"\x00\x01\x02\x7f\x80\xff", b"raw_bytes"]
        for val in bin_vals:
            self.assertIsNone(store.queue_push("bin_q", val))
        self.assertEqual(store.queue_len("bin_q"), len(bin_vals))
        self.assertEqual(
            [store.queue_pop("bin_q", block=False) for _ in bin_vals],
            bin_vals,
        )

    def test_duplicate_values_and_prefix_isolation(self):
        base_store = self._make_tcp_store()
        base_store.queue_push("q1", "dup")
        base_store.queue_push("q1", "dup")
        base_store.queue_push("q2", "other")
        self.assertEqual(base_store.queue_len("q1"), 2)
        self.assertEqual(base_store.queue_len("q2"), 1)
        self.assertEqual(base_store.queue_pop("q1", block=False), b"dup")
        self.assertEqual(base_store.queue_pop("q1", block=False), b"dup")
        self.assertEqual(base_store.queue_pop("q2", block=False), b"other")

        p1, p2 = dist.PrefixStore("ns_one", base_store), dist.PrefixStore("ns_two", base_store)
        p1.queue_push("shared_name", b"val_1")
        p2.queue_push("shared_name", b"val_2")
        self.assertEqual(p1.queue_len("shared_name"), 1)
        self.assertEqual(p2.queue_len("shared_name"), 1)
        self.assertEqual(p1.queue_pop("shared_name", block=False), b"val_1")
        self.assertEqual(p2.queue_pop("shared_name", block=False), b"val_2")

    def test_invalid_args_raises(self):
        store = self._make_tcp_store()
        for args in [("q", 12345), (123, "val"), ("q",)]:
            with self.assertRaises(TypeError):
                store.queue_push(*args)


if __name__ == "__main__":
    run_tests()
