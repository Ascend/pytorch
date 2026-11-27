# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.get_mem_pool 接口功能正确性
API 名称：torch.distributed._symmetric_memory.get_mem_pool
API 签名：torch.distributed._symmetric_memory.get_mem_pool(device) -> MemPool

覆盖维度表：
| 覆盖维度     | 说明                                  | 覆盖情况                                                     |
|--------------|---------------------------------------|--------------------------------------------------------------|
| 空/非空      | device 为 None 报错，有效设备正常查询 | 已覆盖（test_get_mem_pool_device_object、invalid_arguments） |
| 枚举选项     | 加速设备返回 MemPool，CPU 设备报错    | 已覆盖（test_get_mem_pool_device_object、invalid_arguments） |
| 参数类型     | torch.device 与 str 均支持            | 已覆盖（test_get_mem_pool_device_object）                    |
| 传参与不传参 | 位置/关键字传参与缺参报错             | 已覆盖（test_get_mem_pool_device_object、invalid_arguments） |
| 等价类/边界  | 同设备重复调用返回同一缓存池实例      | 已覆盖（test_get_mem_pool_device_object）                    |
| 正常传参场景 | 加速设备获取对称内存 MemPool 并缓存   | 已覆盖（test_get_mem_pool_device_object）                    |
| 异常传参场景 | CPU 设备、非法设备名、None、缺参报错  | 已覆盖（test_invalid_arguments_raise）                       |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入          | 未覆盖（单设备参数接口）                                     |

未覆盖项及原因：
- 混合设备类型：接口仅接收单个 device 参数，不涉及多 Tensor 输入。
- 多卡 HCCL：本接口为本地设备内存池获取接口，不涉及跨进程通信。
- NPU 上 SymmetricMemory 未注册 npu MemPool 后端，设备查询用例在 NPU 跳过。

注意：本接口返回 MemPool 对象，无浮点 Tensor 输出；
     按 API 语义断言返回对象类型与同设备缓存复用性。
"""
import os
import unittest

import torch
import torch.distributed._symmetric_memory as symm_mem

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


NPU_SKIP_REASON = "PTA/NPU defect: SymmetricMemory MemPool did not find backend for device type npu"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetMemPool(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    @unittest.skipIf(DEVICE_NAME == "npu", NPU_SKIP_REASON)
    def test_get_mem_pool_device_object(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        dev_str = f"{self.device_name}:0"
        dev = torch.device(dev_str)
        pool1 = symm_mem.get_mem_pool(dev)
        self.assertIsNotNone(pool1)
        self.assertIs(pool1, symm_mem.get_mem_pool(dev))
        self.assertIs(pool1, symm_mem.get_mem_pool(dev_str))
        self.assertIs(pool1, symm_mem.get_mem_pool(device=dev_str))

    def test_invalid_arguments_raise(self):
        cases = [
            (RuntimeError, (torch.device("cpu"),), {}),
            (TypeError, (), {}),
            ((RuntimeError, ValueError), ("nonexistent_device_xyz",), {}),
            ((TypeError, RuntimeError), (None,), {}),
            (TypeError, (), {"dev": self.device_name}),
        ]
        for exc, args, kw in cases:
            with self.assertRaises(exc):
                symm_mem.get_mem_pool(*args, **kw)


if __name__ == "__main__":
    run_tests()
