# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Store.__init__ 基类构造接口功能正确性
API 名称：torch.distributed.Store.__init__
API 签名：Store.__init__(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 自定义 Python Store 子类构造后为空存储 | 已覆盖（test_python_subclass_store_init） |
| 枚举选项 | 无枚举选项参数 | 未覆盖（无参构造器） |
| 参数类型 | 无参构造与非法位置/关键字参数 | 已覆盖（test_base_store_init_and_unimplemented、test_invalid_args） |
| 传参与不传参 | 仅支持无参构造；多余参数抛 TypeError | 已覆盖（test_base_store_init_and_unimplemented、test_invalid_args） |
| 等价类/边界值 | 裸基类与子类/PrefixStore | 已覆盖（test_base_store_init_and_unimplemented、test_python_subclass_store_init） |
| 正常传参场景 | Python 子类调用 super().__init__() 并完成键值读写 | 已覆盖（test_python_subclass_store_init） |
| 异常传参场景 | 裸基类调用纯虚方法抛 RuntimeError；非法入参抛 TypeError | 已覆盖（test_base_store_init_and_unimplemented、test_invalid_args） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（键值存储基类构造器，无 Tensor 参数） |

未覆盖项及原因：
- 枚举选项：Store.__init__ 为无参构造器，无枚举参数。
- 混合设备类型：接口不接收 Tensor 参数，不涉及跨设备输入。
- 多卡 HCCL 通信：依据分布式规范，Store 系单进程测试。

注意：Store 为 C++ 抽象基类（含 pybind11 trampoline），直接实例化后调用未实现虚方法会抛 RuntimeError；
     通过 Python 子类重写核心方法并调用 super().__init__() 可完整验证基类构造语义。
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


class _InMemoryPythonStore(dist.Store):
    def __init__(self):
        super().__init__()
        self._kv = {}

    def set(self, key, value):
        self._kv[key] = value.encode("utf-8") if isinstance(value, str) else bytes(value)

    def get(self, key):
        return self._kv[key]

    def check(self, keys):
        return all(k in self._kv for k in keys)

    def add(self, key, value):
        nxt = int(self._kv.get(key, b"0").decode("utf-8")) + value
        self._kv[key] = str(nxt).encode("utf-8")
        return nxt

    def delete_key(self, key):
        return self._kv.pop(key, None) is not None

    def num_keys(self):
        return len(self._kv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestStoreInit(TestCase):
    def test_base_store_init_and_unimplemented(self):
        base = dist.Store()
        self.assertIsInstance(base, dist.Store)
        with self.assertRaises(RuntimeError):
            base.get("missing")

    def test_python_subclass_store_init(self):
        self.assertIsInstance(dist.HashStore(), dist.Store)
        store = _InMemoryPythonStore()
        self.assertIsInstance(store, dist.Store)
        self.assertEqual(store.num_keys(), 0)
        self.assertFalse(store.check(["k1"]))

        store.set("k1", "v1")
        self.assertTrue(store.check(["k1"]))
        self.assertEqual(store.get("k1"), b"v1")
        self.assertEqual(store.add("cnt", 4), 4)
        self.assertEqual(store.num_keys(), 2)
        self.assertTrue(store.delete_key("k1"))
        self.assertFalse(store.check(["k1"]))

        prefixed = dist.PrefixStore("py", store)
        prefixed.set("item", "hello")
        self.assertEqual(prefixed.get("item"), b"hello")
        self.assertEqual(store.get("py/item"), b"hello")

    def test_invalid_args(self):
        for args, kwargs in [((123,), {}), ((), {"timeout": 5})]:
            with self.assertRaises(TypeError):
                dist.Store(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
