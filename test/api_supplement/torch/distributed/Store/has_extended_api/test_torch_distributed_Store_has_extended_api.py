# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.has_extended_api 接口功能正确性
API 名称：torch.distributed.Store.has_extended_api
API 签名：Store.has_extended_api(self) -> bool

覆盖维度表：
| 覆盖维度     | 说明                                       | 覆盖情况                                           |
|--------------|--------------------------------------------|----------------------------------------------------|
| 空/非空      | 不涉及容器或空 Tensor 输入                 | 未覆盖（无参布尔能力查询接口）                     |
| 枚举选项     | TCPStore(libuv)/HashStore/FileStore/Prefix | 已覆盖（test_tcp_store_libuv_true_and_false、      |
|              |                                            | test_hash_file_and_prefix_store_capability）       |
| 参数类型     | 无参调用与非法多余参数                     | 已覆盖（test_invalid_extra_args_raises）           |
| 传参与不传参 | 无额外参数，验证默认无参调用               | 已覆盖（全部正常用例）                             |
| 等价类/边界  | 返回 True 时扩展接口可用，False 时委托一致 | 已覆盖（test_tcp_store_libuv_true_and_false、      |
|              |                                            | test_hash_file_and_prefix_store_capability）       |
| 正常传参场景 | 不同 Store 实现返回确定布尔结果            | 已覆盖（test_tcp_store_libuv_true_and_false、      |
|              |                                            | test_hash_file_and_prefix_store_capability）       |
| 异常传参场景 | 传入多余位置参数触发 TypeError             | 已覆盖（test_invalid_extra_args_raises）           |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 能力查询接口不接收 Tensor 参数）     |

未覆盖项及原因：
- 空/非空：接口为无参布尔能力查询方法，无容器或 Tensor 输入。
- 混合设备类型：接口不接收 Tensor 参数，不涉及跨设备 Tensor 混合输入。

注意：本接口查询 Store 实现是否支持扩展 API（如 list_keys、queue_*），无浮点 Tensor 输出；
     针对不同 Store 子类断言确定的 True/False 布尔结果并验证扩展接口行为一致性。
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
class TestStoreHasExtendedApi(TestCase):
    def _make_tcp_store(self, use_libuv=True):
        return dist.TCPStore(
            "127.0.0.1", 0, 1, is_master=True, wait_for_workers=False, use_libuv=use_libuv
        )

    def _run_with_file_store(self, fn):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        try:
            fstore = dist.FileStore(path, 1)
            try:
                fn(fstore)
            finally:
                del fstore
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_tcp_store_libuv_true_and_false(self):
        tcp_libuv = self._make_tcp_store(use_libuv=True)
        self.assertIs(tcp_libuv.has_extended_api(), True)
        tcp_libuv.set("probe_key", "probe_val")
        self.assertIn("probe_key", tcp_libuv.list_keys())

        tcp_legacy = self._make_tcp_store(use_libuv=False)
        self.assertIs(tcp_legacy.has_extended_api(), True)

    def test_hash_file_and_prefix_store_capability(self):
        self.assertIs(dist.HashStore().has_extended_api(), True)
        prefix_tcp = dist.PrefixStore("pref", self._make_tcp_store(use_libuv=True))
        self.assertIs(prefix_tcp.has_extended_api(), True)

        def _check_file_store(fs):
            self.assertIs(fs.has_extended_api(), False)
            self.assertIs(dist.PrefixStore("pref", fs).has_extended_api(), False)

        self._run_with_file_store(_check_file_store)

    def test_invalid_extra_args_raises(self):
        with self.assertRaises(TypeError):
            dist.HashStore().has_extended_api(True)


if __name__ == "__main__":
    run_tests()
