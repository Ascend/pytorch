# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.check 非阻塞键存在性检查接口功能正确性
API 名称：torch.distributed.Store.check
API 签名：Store.check(self, keys: list[str]) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空键列表 [] 返回 True；非空键列表存在/缺失 | 已覆盖（test_empty_and_multiple_keys_lifecycle） |
| 枚举选项 | 无枚举选项参数 | 未覆盖（Store.check 无枚举入参） |
| 参数类型 | list[str] 合法入参与非列表/非字符串元素异常 | 已覆盖（test_empty_and_multiple_keys_lifecycle、test_invalid_args） |
| 传参与不传参 | 唯一必填参数 keys；缺参抛 TypeError | 已覆盖（test_invalid_args） |
| 等价类/边界值 | 全存在、部分存在、全缺失、重复键、删键后变 False | 已覆盖（test_empty_and_multiple_keys_lifecycle） |
| 正常传参场景 | HashStore 与 PrefixStore 非阻塞检查 | 已覆盖（test_empty_and_multiple_keys_lifecycle、test_prefix_store_check） |
| 异常传参场景 | 缺参、传 str 或非 str 元素列表抛 TypeError | 已覆盖（test_invalid_args） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（键值存储查询方法，无 Tensor 参数） |

未覆盖项及原因：
- 枚举选项：Store.check 仅接受 keys 列表，无枚举参数。
- 混合设备类型：接口不接收 Tensor 参数，不涉及跨设备输入。
- 多卡 HCCL 通信：依据分布式规范，Store 系单进程测试。

注意：Store.check 为非阻塞存在性探测，对缺失键立即返回 False 而不会阻塞等待。
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
class TestStoreCheck(TestCase):
    def test_empty_and_multiple_keys_lifecycle(self):
        store = dist.HashStore()
        self.assertIs(store.check([]), True)
        self.assertIs(store.check(["missing"]), False)

        store.set("a", "1")
        store.set("b", "2")
        self.assertIs(store.check(["a"]), True)
        self.assertIs(store.check(["a", "a"]), True)
        self.assertIs(store.check(["a", "b"]), True)
        self.assertIs(store.check(["a", "missing"]), False)
        self.assertIs(store.check(["missing1", "missing2"]), False)

        self.assertTrue(store.delete_key("a"))
        self.assertIs(store.check(["a"]), False)
        self.assertIs(store.check(["a", "b"]), False)
        self.assertIs(store.check(["b"]), True)

    def test_prefix_store_check(self):
        base = dist.HashStore()
        p1, p2 = dist.PrefixStore("ns1", base), dist.PrefixStore("ns2", base)
        p1.set("shared", "v")
        self.assertIs(p1.check(["shared"]), True)
        self.assertIs(p2.check(["shared"]), False)
        self.assertIs(base.check(["ns1/shared"]), True)
        self.assertIs(base.check(["ns2/shared"]), False)

    def test_invalid_args(self):
        store = dist.HashStore()
        for args in [(), ("not_a_list",), ([123],)]:
            with self.assertRaises(TypeError):
                store.check(*args)


if __name__ == "__main__":
    run_tests()
