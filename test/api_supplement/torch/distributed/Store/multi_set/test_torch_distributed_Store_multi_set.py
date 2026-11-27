# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.multi_set 接口功能正确性
API 名称：torch.distributed.Store.multi_set
API 签名：Store.multi_set(self, keys: list[str], values: list[str | bytes]) -> None

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 空列表调用、单元素列表、多元素批量写入     | 已覆盖（test_empty_str_bytes_and_overwrite）           |
| 枚举选项       | TCPStore/HashStore/FileStore/PrefixStore   | 已覆盖（test_empty_str_bytes_and_overwrite、           |
|                |                                            | test_hash_file_and_prefix_stores）                     |
| 参数类型       | str 与 bytes 值（含二进制零字节）、非法类型 | 已覆盖（test_empty_str_bytes_and_overwrite、           |
|                |                                            | test_invalid_args_and_length_mismatch）                |
| 传参与不传参   | keys 与 values 均为必选参数                | 已覆盖（全部正常用例与缺参异常用例）                   |
| 等价类/边界值  | 空值、二进制字节、重复键覆盖、批量写入     | 已覆盖（test_empty_str_bytes_and_overwrite）           |
| 正常传参场景   | multi_set 后用 get/multi_get 逐项校验      | 已覆盖（全部正常用例）                                 |
| 异常传参场景   | keys/values 长度不一致、类型错误、缺参     | 已覆盖（test_invalid_args_and_length_mismatch）        |
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 键值写入接口不接收 Tensor 参数）         |

未覆盖项及原因：
- 混合设备类型：Store.multi_set 仅批量写入字符串/字节键值对，不接收 Tensor 参数。

注意：本接口为分布式 Store 批量键值写入方法，无浮点 Tensor 输出；
     写入后通过 get / multi_get 逐项比对字节内容验证持久化正确性。
"""
import os
import tempfile
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
class TestStoreMultiSet(TestCase):
    def _make_tcp_store(self):
        return dist.TCPStore(
            "127.0.0.1", 0, 1, is_master=True, wait_for_workers=False, use_libuv=True
        )

    def test_empty_str_bytes_and_overwrite(self):
        store = self._make_tcp_store()
        self.assertIsNone(store.multi_set([], []))
        self.assertEqual(store.list_keys(), [])

        self.assertIsNone(store.multi_set(["only_key"], [""]))
        self.assertEqual(store.get("only_key"), b"")

        keys = ["k1", "k2", "k3"]
        str_vals = ["val_one", "val_two", "unicode_\u4e2d\u6587"]
        self.assertIsNone(store.multi_set(keys, str_vals))
        self.assertEqual(
            store.multi_get(keys),
            [b"val_one", b"val_two", "unicode_\u4e2d\u6587".encode("utf-8")],
        )

        bin_keys = ["b1", "b2"]
        bin_vals = [b"\x00\x01\x02\xff", b""]
        self.assertIsNone(store.multi_set(bin_keys, bin_vals))
        self.assertEqual(store.multi_get(bin_keys), bin_vals)

        store.multi_set(["k1", "k4"], ["new_1", "new_4"])
        self.assertEqual(store.get("k2"), b"val_two")
        self.assertEqual(store.multi_get(["k1", "k4"]), [b"new_1", b"new_4"])

    def test_hash_file_and_prefix_stores(self):
        for store, keys, vals in [
            (dist.HashStore(), ["hk1", "hk2"], ["hv1", "hv2"]),
            (dist.PrefixStore("ns", self._make_tcp_store()), ["pk1", "pk2"], ["pv1", "pv2"]),
        ]:
            store.multi_set(keys, vals)
            self.assertEqual(store.multi_get(keys), [v.encode("utf-8") for v in vals])

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file_path = tmp.name
        try:
            file_store = dist.FileStore(file_path, 1)
            file_store.multi_set(["fk1", "fk2"], ["fv1", "fv2"])
            self.assertEqual(file_store.multi_get(["fk1", "fk2"]), [b"fv1", b"fv2"])
            del file_store
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_invalid_args_and_length_mismatch(self):
        store = self._make_tcp_store()
        with self.assertRaises((RuntimeError, ValueError)):
            store.multi_set(["k1", "k2"], ["v1"])
        for args in [("not_a_list", ["v1"]), (["k1"],)]:
            with self.assertRaises(TypeError):
                store.multi_set(*args)


if __name__ == "__main__":
    run_tests()
