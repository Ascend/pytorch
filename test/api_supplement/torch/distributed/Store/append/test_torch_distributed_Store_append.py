# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.append 接口功能正确性
API 名称：torch.distributed.Store.append
API 签名：Store.append(self, key: str, value: str | bytes) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空字符串/空字节追加与非空值连续追加 | 已覆盖（test_append_str_bytes_and_empty） |
| 枚举选项 | 无枚举选项参数 | 未覆盖（Store.append 无枚举入参） |
| 参数类型 | str 与 bytes（含二进制 NUL 字节）混合追加 | 已覆盖（test_append_str_bytes_and_empty、test_invalid_args_and_unimplemented） |
| 传参与不传参 | 两个必填位置参数；缺参抛 TypeError | 已覆盖（test_invalid_args_and_unimplemented） |
| 等价类/边界值 | 新建键首次追加、已有键多次追加、PrefixStore 透传 | 已覆盖（test_append_str_bytes_and_empty、test_prefix_store_append） |
| 正常传参场景 | HashStore 与 PrefixStore 上追加并校验完整字节序列 | 已覆盖（test_append_str_bytes_and_empty、test_prefix_store_append） |
| 异常传参场景 | 非法类型/缺参抛 TypeError；未实现子类抛 RuntimeError | 已覆盖（test_invalid_args_and_unimplemented） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（键值存储方法，无 Tensor 参数） |

未覆盖项及原因：
- 枚举选项：Store.append 仅接受 key 与 value，无枚举参数。
- 混合设备类型：接口不接收 Tensor 参数，不涉及跨设备输入。
- 多卡 HCCL 通信：依据分布式规范，Store 系单进程测试。

注意：HashStore 与 PrefixStore(HashStore) 实现扩展 API append；
     裸 Store 基类未实现 append，调用会抛出 RuntimeError。
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
class TestStoreAppend(TestCase):
    def test_append_str_bytes_and_empty(self):
        store = dist.HashStore()
        self.assertFalse(store.check(["new_key"]))
        store.append("new_key", "")
        self.assertTrue(store.check(["new_key"]))
        self.assertEqual(store.get("new_key"), b"")
        store.append("new_key", "first")
        self.assertEqual(store.get("new_key"), b"first")

        store.append("empty_bytes_key", b"")
        self.assertEqual(store.get("empty_bytes_key"), b"")

        store.set("log", "init")
        for chunk in ("-part1", "-part2", ""):
            store.append("log", chunk)
        self.assertEqual(store.get("log"), b"init-part1-part2")

        for chunk in (b"\x00\x01", "AB", b"\xfe\xff"):
            store.append("bin", chunk)
        self.assertEqual(store.get("bin"), b"\x00\x01AB\xfe\xff")

    def test_prefix_store_append(self):
        base = dist.HashStore()
        prefixed = dist.PrefixStore("pfx", base)
        prefixed.append("seq", "a")
        prefixed.append("seq", "b")
        self.assertEqual(prefixed.get("seq"), b"ab")
        self.assertEqual(base.get("pfx/seq"), b"ab")

    def test_invalid_args_and_unimplemented(self):
        with self.assertRaises(RuntimeError):
            dist.Store().append("k", "v")
        store = dist.HashStore()
        for args in [("only_key",), (123, "val"), ("key", 456)]:
            with self.assertRaises(TypeError):
                store.append(*args)


if __name__ == "__main__":
    run_tests()
