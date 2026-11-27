# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.clone 接口功能正确性
API 名称：torch.distributed.Store.clone
API 签名：Store.clone(self) -> Store

覆盖维度表：
| 覆盖维度       | 说明                                       | 覆盖情况                                               |
|----------------|--------------------------------------------|--------------------------------------------------------|
| 空/非空        | 空 Store 与非空 Store 克隆                 | 已覆盖（test_tcp_store_clone_empty_and_shared_state）  |
| 枚举选项       | TCPStore 与 PrefixStore 两种支持克隆的子类 | 已覆盖（test_tcp_store_clone_empty_and_shared_state、  |
|                |                                            | test_prefix_store_clone_preserves_prefix）             |
| 参数类型       | 无参调用与非法多余参数                     | 已覆盖（test_invalid_extra_args_raises）               |
| 传参与不传参   | 无额外参数，验证默认无参调用               | 已覆盖（全部正常用例）                                 |
| 等价类/边界值  | 独立超时配置不影响共享数据                 | 已覆盖（test_tcp_store_clone_empty_and_shared_state）  |
| 正常传参场景   | 双向读写同步、删除同步、PrefixStore 隔离   | 已覆盖（test_tcp_store_clone_empty_and_shared_state、  |
|                |                                            | test_prefix_store_clone_preserves_prefix）             |
| 异常传参场景   | 传入多余位置参数触发 TypeError             | 已覆盖（test_invalid_extra_args_raises）               |
| 混合设备类型   | 不涉及多 Tensor 混合设备输入               | 未覆盖（Store 键值协调接口不接收 Tensor 参数）         |

未覆盖项及原因：
- 混合设备类型：Store.clone 仅克隆分布式键值存储客户端句柄，不接收 Tensor 参数。

注意：本接口为分布式键值存储客户端克隆方法，属于纯单进程协调接口，无浮点 Tensor 输出；
     使用对象身份、键值读写一致性、删除可见性与异常断言进行验证。
"""
import datetime
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
class TestStoreClone(TestCase):
    def _make_tcp_store(self):
        return dist.TCPStore("127.0.0.1", 0, 1, is_master=True, wait_for_workers=False)

    def test_tcp_store_clone_empty_and_shared_state(self):
        store = self._make_tcp_store()
        cloned = store.clone()

        self.assertIsInstance(cloned, dist.Store)
        self.assertIsNot(cloned, store)
        self.assertFalse(cloned.check(["k_orig"]))

        store.set("k_orig", "v_orig")
        self.assertTrue(cloned.check(["k_orig"]))
        self.assertEqual(cloned.get("k_orig"), b"v_orig")

        cloned.set("k_clone", "v_clone")
        self.assertTrue(store.check(["k_clone"]))
        self.assertEqual(store.get("k_clone"), b"v_clone")

        self.assertTrue(cloned.delete_key("k_orig"))
        self.assertFalse(store.check(["k_orig"]))

        store.set_timeout(datetime.timedelta(seconds=7))
        cloned.set_timeout(datetime.timedelta(seconds=13))
        self.assertEqual(store.timeout, datetime.timedelta(seconds=7))
        self.assertEqual(cloned.timeout, datetime.timedelta(seconds=13))

    def test_prefix_store_clone_preserves_prefix(self):
        base_store = self._make_tcp_store()
        prefix_a = dist.PrefixStore("ns_a", base_store)
        prefix_b = dist.PrefixStore("ns_b", base_store)

        cloned_a = prefix_a.clone()
        self.assertIsInstance(cloned_a, dist.Store)
        self.assertIsNot(cloned_a, prefix_a)

        prefix_a.set("item", "from_a")
        prefix_b.set("item", "from_b")

        self.assertEqual(cloned_a.get("item"), b"from_a")
        cloned_a.set("extra", "from_cloned_a")
        self.assertEqual(prefix_a.get("extra"), b"from_cloned_a")
        self.assertFalse(prefix_b.check(["extra"]))

    def test_invalid_extra_args_raises(self):
        with self.assertRaises(TypeError):
            self._make_tcp_store().clone(1)


if __name__ == "__main__":
    run_tests()
