# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.get_backend 接口功能正确性
API 名称：torch.distributed._symmetric_memory.get_backend
API 签名：torch.distributed._symmetric_memory.get_backend(device) -> str | None

覆盖维度表：
| 覆盖维度     | 说明                                  | 覆盖情况                                                      |
|--------------|---------------------------------------|---------------------------------------------------------------|
| 空/非空      | device 为 None 报错，有效设备正常查询 | 已覆盖（test_device_object_and_string、invalid_arguments）    |
| 枚举选项     | 加速设备返回后端名，CPU 返回 None     | 已覆盖（test_device_object_and_string）                       |
| 参数类型     | torch.device 与 str 均支持            | 已覆盖（test_device_object_and_string）                       |
| 传参与不传参 | 位置/关键字传参与缺参报错             | 已覆盖（test_device_object_and_string、invalid_arguments）    |
| 等价类/边界  | 带索引与不带索引设备等价              | 已覆盖（test_device_object_and_string）                       |
| 正常传参场景 | NPU/CUDA 返回后端名（NPU 为 NPUSHMEM）| 已覆盖（test_device_object_and_string）                       |
| 异常传参场景 | 缺参、非法设备名、None、未知关键字    | 已覆盖（test_invalid_arguments_raise）                        |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入          | 未覆盖（单设备参数查询接口）                                  |

未覆盖项及原因：
- 混合设备类型：接口仅接收单个 device 参数，不涉及多 Tensor 输入。
- 多卡 HCCL：本接口为本地后端注册查询，不涉及跨进程通信。

注意：本接口返回后端名称字符串或 None，无浮点 Tensor 输出；
     在 NPU 上断言返回 "NPUSHMEM"，在 CPU 设备上断言返回 None。
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGetBackend(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_device_object_and_string(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        indexed = f"{self.device_name}:0"
        indexed_dev = torch.device(indexed)
        results = [
            symm_mem.get_backend(d)
            for d in (indexed_dev, torch.device(self.device_name), indexed, self.device_name)
        ]
        self.assertIsInstance(results[0], str)
        self.assertTrue(len(results[0]) > 0)
        for res in results[1:]:
            self.assertEqual(results[0], res)
        self.assertEqual(symm_mem.get_backend(device=indexed_dev), results[0])
        if self.device_name == "npu":
            self.assertEqual(results[0], "NPUSHMEM")
        for dev in (torch.device("cpu"), "cpu"):
            self.assertIsNone(symm_mem.get_backend(dev))

    def test_invalid_arguments_raise(self):
        cases = [
            (TypeError, (), {}),
            ((RuntimeError, ValueError), ("nonexistent_device_xyz",), {}),
            ((TypeError, RuntimeError), (None,), {}),
            (TypeError, (), {"dev": self.device_name}),
        ]
        for exc, args, kw in cases:
            with self.assertRaises(exc):
                symm_mem.get_backend(*args, **kw)


if __name__ == "__main__":
    run_tests()
