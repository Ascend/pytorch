# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.HashStore.__init__ 构造接口功能正确性
API 名称：torch.distributed.HashStore.__init__
API 签名：HashStore.__init__(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 构造后初始存储为空（num_keys == 0） | 已覆盖（test_default_init_and_isolation） |
| 枚举选项 | 无枚举选项参数 | 未覆盖（无参构造器） |
| 参数类型 | 无参调用与非法位置/关键字参数 | 已覆盖（test_default_init_and_isolation、test_invalid_arguments_raise） |
| 传参与不传参 | 仅支持无参构造；多余入参报错 | 已覆盖（test_default_init_and_isolation、test_invalid_arguments_raise） |
| 等价类/边界值 | 多实例状态隔离、Python 子类 super().__init__ | 已覆盖（test_default_init_and_isolation、test_subclass_super_init） |
| 正常传参场景 | 构造后可执行 set/get/check/add | 已覆盖（test_default_init_and_isolation、test_subclass_super_init） |
| 异常传参场景 | 传入多余位置参数或关键字参数抛 TypeError | 已覆盖（test_invalid_arguments_raise） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（无参内存存储构造器） |

未覆盖项及原因：
- 枚举选项：HashStore.__init__ 为无参构造器，无枚举参数。
- 混合设备类型：接口不接收 Tensor 参数，不涉及设备迁移或混合设备输入。
- 多卡 HCCL 通信：依据分布式规范，Store 系单进程测试。

注意：使用非阻塞 check([key]) 校验实例间键隔离，避免对缺失键调用阻塞 get()。
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
class TestHashStoreInit(TestCase):

    def test_default_init_and_isolation(self):
        store_a, store_b = dist.HashStore(), dist.HashStore()
        self.assertIsInstance(store_a, dist.HashStore)
        self.assertIsInstance(store_a, dist.Store)
        self.assertEqual(store_a.num_keys(), 0)
        self.assertFalse(store_a.check(["shared_key"]))

        store_a.set("shared_key", "from_a")
        self.assertEqual(store_a.num_keys(), 1)
        self.assertTrue(store_a.check(["shared_key"]))
        self.assertFalse(store_b.check(["shared_key"]))
        self.assertEqual(store_b.num_keys(), 0)

        store_b.set("shared_key", "from_b")
        self.assertEqual(store_a.get("shared_key"), b"from_a")
        self.assertEqual(store_b.get("shared_key"), b"from_b")

    def test_subclass_super_init(self):
        class _SubHashStore(dist.HashStore):
            def __init__(self, tag):
                super().__init__()
                self.tag = tag

        sub = _SubHashStore("custom_tag")
        self.assertEqual(sub.tag, "custom_tag")
        self.assertIsInstance(sub, dist.HashStore)
        self.assertEqual(sub.num_keys(), 0)
        sub.set("sub_key", "sub_val")
        self.assertEqual(sub.get("sub_key"), b"sub_val")
        self.assertEqual(sub.add("sub_cnt", 3), 3)

    def test_invalid_arguments_raise(self):
        for args, kwargs in (((1,), {}), ((), {"timeout": 10}), (("a", "b"), {})):
            with self.assertRaises(TypeError):
                dist.HashStore(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
