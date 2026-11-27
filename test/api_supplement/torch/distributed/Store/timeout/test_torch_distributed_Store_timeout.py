# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.timeout 属性功能正确性
API 名称：torch.distributed.Store.timeout
API 签名：Store.timeout (property: datetime.timedelta)

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 不涉及容器或空 Tensor 输入                 | 未覆盖（标量 timedelta 属性读取）                      |
| 枚举选项       | TCPStore/HashStore/FileStore/PrefixStore   | 已覆盖（test_default_constructor_and_set_timeout_sync）|
| 参数类型       | datetime.timedelta 与非法参数类型          | 已覆盖（test_default_constructor_and_set_timeout_sync、|
|                |                                            | test_invalid_timeout_type_raises）                     |
| 传参与不传参   | 构造默认 timeout vs 显式传入 timeout       | 已覆盖（test_default_constructor_and_set_timeout_sync）|
| 等价类/边界值  | 毫秒级短超时真实生效于 wait 超时抛错       | 已覆盖（test_short_timeout_takes_effect_on_wait）      |
| 正常传参场景   | 读取属性并与 set_timeout 联动、Prefix 委托 | 已覆盖（全部正常用例）                                 |
| 异常传参场景   | 只读属性赋值抛 AttributeError、错参 TypeError | 已覆盖（test_invalid_timeout_type_raises）          |
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 超时属性不接收 Tensor 参数）             |

未覆盖项及原因：
- 空/非空：timeout 为单个 datetime.timedelta 属性，不涉及容器或 Tensor 输入。
- 混合设备类型：Store.timeout 不接收 Tensor 参数，不涉及跨设备 Tensor 混合输入。

注意：本接口为分布式 Store 超时属性，无浮点 Tensor 输出；
     通过精确 timedelta 值比对、与 set_timeout 联动、wait 真实超时生效及异常断言验证。
"""
import datetime
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
class TestStoreTimeout(TestCase):
    def test_default_constructor_and_set_timeout_sync(self):
        hash_store = dist.HashStore()
        self.assertIsInstance(hash_store.timeout, datetime.timedelta)
        self.assertGreater(hash_store.timeout, datetime.timedelta(0))

        custom_td = datetime.timedelta(seconds=15)
        tcp_store = dist.TCPStore(
            "127.0.0.1", 0, 1, is_master=True, timeout=custom_td, wait_for_workers=False
        )
        self.assertEqual(tcp_store.timeout, custom_td)

        for td in (datetime.timedelta(seconds=4), datetime.timedelta(milliseconds=250)):
            tcp_store.set_timeout(td)
            self.assertEqual(tcp_store.timeout, td)

        prefix_store = dist.PrefixStore("ns", hash_store)
        delegated_td = datetime.timedelta(seconds=9)
        prefix_store.set_timeout(delegated_td)
        self.assertEqual(prefix_store.timeout, delegated_td)
        self.assertEqual(hash_store.timeout, delegated_td)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file_path = tmp.name
        try:
            file_store = dist.FileStore(file_path, 1)
            self.assertIsInstance(file_store.timeout, datetime.timedelta)
            self.assertGreater(file_store.timeout, datetime.timedelta(0))
            del file_store
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_short_timeout_takes_effect_on_wait(self):
        store = dist.HashStore()
        td = datetime.timedelta(milliseconds=100)
        store.set_timeout(td)
        self.assertEqual(store.timeout, td)
        with self.assertRaises(RuntimeError):
            store.wait(["nonexistent_key_for_timeout_test"])

    def test_invalid_timeout_type_raises(self):
        store = dist.HashStore()
        for bad_val in (datetime.timedelta(seconds=5), 10):
            with self.assertRaises(AttributeError):
                store.timeout = bad_val
        with self.assertRaises(TypeError):
            store.set_timeout("5s")


if __name__ == "__main__":
    run_tests()
