# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.HashStore 类功能正确性
API 名称：torch.distributed.HashStore
API 签名：class torch.distributed.HashStore(Store)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空库 num_keys=0、空字符串/空字节值与非空值 | 已覆盖（test_basic_kv_and_numeric_ops） |
| 枚举选项 | 无离散枚举参数 | 未覆盖（HashStore 无枚举入参） |
| 参数类型 | str 与 bytes 键值、整数计数、timedelta 超时 | 已覆盖（test_basic_kv_and_numeric_ops、test_wait_and_timeout） |
| 传参与不传参 | wait 默认超时与显式 timedelta 超时 | 已覆盖（test_wait_and_timeout） |
| 等价类/边界值 | 覆盖写、负增量、二进制含 NUL 字节、缺键 check | 已覆盖（test_basic_kv_and_numeric_ops） |
| 正常传参场景 | set/get/add/check/compare_set/delete/wait | 已覆盖（test_basic_kv_and_numeric_ops、test_wait_and_timeout） |
| 异常传参场景 | wait 缺键超时抛 RuntimeError、非法构造参数抛 TypeError | 已覆盖（test_wait_and_timeout、test_invalid_args） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯键值存储类，无 Tensor 参数） |

未覆盖项及原因：
- 枚举选项：HashStore 类方法不涉及枚举选项参数。
- 混合设备类型：HashStore 为内存键值存储，不接收 Tensor 参数。
- 多卡 HCCL 通信：依据分布式规范，Store 系单进程测试，不依赖多卡通信。

注意：缺键探测使用非阻塞 check([key])，不对缺失 key 直接调用无超时阻塞 get()；
     测试超时路径时显式设置短超时（毫秒级）以快速验证异常。
"""
from datetime import timedelta
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
class TestHashStore(TestCase):

    def test_basic_kv_and_numeric_ops(self):
        self.assertTrue(issubclass(dist.HashStore, dist.Store))
        store = dist.HashStore()
        self.assertIsInstance(store, dist.HashStore)
        self.assertIsInstance(store, dist.Store)
        self.assertEqual(store.num_keys(), 0)
        self.assertFalse(store.check(["missing_key"]))

        store.set("empty_str", "")
        store.set("empty_bytes", b"")
        self.assertEqual(store.get("empty_str"), b"")
        self.assertEqual(store.get("empty_bytes"), b"")

        store.set("k1", "value1")
        store.set("k1", "value1_updated")
        raw = b"\x00\x01\x7f\x80\xff"
        store.set("bin_key", raw)
        self.assertEqual(store.multi_get(["k1", "bin_key"]), [b"value1_updated", raw])
        self.assertTrue(store.check(["k1", "bin_key"]))

        for delta, expected in ((1, 1), (5, 6), (-2, 4), (0, 4)):
            self.assertEqual(store.add("counter", delta), expected)

        # compare_set: empty expected initializes absent key; mismatch preserves; match updates.
        self.assertEqual(store.compare_set("cas_key", "", "v1"), b"v1")
        self.assertEqual(store.compare_set("cas_key", "wrong", "v2"), b"v1")
        self.assertEqual(store.compare_set("cas_key", "v1", "v2"), b"v2")
        self.assertTrue(store.delete_key("cas_key"))
        self.assertFalse(store.delete_key("cas_key"))

        prefixed = dist.PrefixStore("ns", store)
        prefixed.set("k", "val")
        self.assertEqual(prefixed.get("k"), b"val")
        self.assertEqual(store.get("ns/k"), b"val")

    def test_wait_and_timeout(self):
        store = dist.HashStore()
        store.set("ready", "yes")
        store.wait(["ready"])
        store.wait(["ready"], timedelta(seconds=1))

        store.set_timeout(timedelta(milliseconds=50))
        with self.assertRaises(RuntimeError):
            store.wait(["never_set_key"], timedelta(milliseconds=50))

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            dist.HashStore("unexpected_arg")
        store = dist.HashStore()
        with self.assertRaises(TypeError):
            store.set(123, "val")
        with self.assertRaises(TypeError):
            store.add("counter", "not_an_int")


if __name__ == "__main__":
    run_tests()
