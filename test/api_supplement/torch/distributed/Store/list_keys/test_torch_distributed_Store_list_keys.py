# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.list_keys 接口功能正确性
API 名称：torch.distributed.Store.list_keys
API 签名：Store.list_keys(self) -> list[str]

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 空 Store 返回空列表，写入后返回完整键列表  | 已覆盖（test_empty_set_overwrite_and_delete_keys）     |
| 枚举选项       | TCPStore 与 PrefixStore 两种场景           | 已覆盖（test_empty_set_overwrite_and_delete_keys、     |
|                |                                            | test_prefix_store_list_keys）                          |
| 参数类型       | 无参调用与非法多余参数                     | 已覆盖（test_invalid_extra_args_raises）               |
| 传参与不传参   | 无额外参数，验证默认无参调用               | 已覆盖（全部正常用例）                                 |
| 等价类/边界值  | 空表、覆盖不重复、删除移除、特殊键名       | 已覆盖（test_empty_set_overwrite_and_delete_keys）     |
| 正常传参场景   | set/add/multi_set/delete_key 后精确校验    | 已覆盖（test_empty_set_overwrite_and_delete_keys、     |
|                |                                            | test_prefix_store_list_keys）                          |
| 异常传参场景   | 传入多余位置参数触发 TypeError             | 已覆盖（test_invalid_extra_args_raises）               |
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 键枚举接口不接收 Tensor 参数）           |

未覆盖项及原因：
- 混合设备类型：Store.list_keys 仅枚举键值存储中的键名，不接收 Tensor 参数。

注意：本接口属于分布式 Store 扩展 API，在单进程 TCPStore（libuv）上验证，无浮点 Tensor 输出；
     通过精确比较键集合、去重与删除后的变化验证功能正确性。
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
class TestStoreListKeys(TestCase):
    def _make_tcp_store(self):
        return dist.TCPStore(
            "127.0.0.1", 0, 1, is_master=True, wait_for_workers=False, use_libuv=True
        )

    def test_empty_set_overwrite_and_delete_keys(self):
        store = self._make_tcp_store()
        empty_keys = store.list_keys()
        self.assertIsInstance(empty_keys, list)
        self.assertEqual(empty_keys, [])

        for k, v in [("alpha", "1"), ("beta", "2"), ("dir/sub/key", "")]:
            store.set(k, v)
        store.add("counter_key", 5)
        store.multi_set(["batch_a"], ["va"])
        store.set("beta", "2_updated")

        expected = ["alpha", "batch_a", "beta", "counter_key", "dir/sub/key"]
        keys = store.list_keys()
        self.assertEqual(len(keys), len(expected))
        self.assertEqual(sorted(keys), expected)

        self.assertTrue(store.delete_key("beta"))
        expected.remove("beta")
        self.assertEqual(sorted(store.list_keys()), expected)

    def test_prefix_store_list_keys(self):
        prefix_store = dist.PrefixStore("my_ns", self._make_tcp_store())
        prefix_store.set("k1", "v1")
        prefix_store.set("k2", "v2")

        keys = prefix_store.list_keys()
        self.assertEqual(len(keys), 2)
        self.assertTrue(any(k.endswith("k1") for k in keys))
        self.assertTrue(any(k.endswith("k2") for k in keys))

        prefix_store.delete_key("k1")
        keys_after = prefix_store.list_keys()
        self.assertEqual(len(keys_after), 1)
        self.assertTrue(any(k.endswith("k2") for k in keys_after))

    def test_invalid_extra_args_raises(self):
        with self.assertRaises(TypeError):
            self._make_tcp_store().list_keys("unexpected")


if __name__ == "__main__":
    run_tests()
