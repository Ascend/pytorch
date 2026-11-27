# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.is_nvshmem_available 接口功能正确性
API 名称：torch.distributed._symmetric_memory.is_nvshmem_available
API 签名：torch.distributed._symmetric_memory.is_nvshmem_available() -> bool

覆盖维度表：
| 覆盖维度     | 说明                             | 覆盖情况                                                       |
|--------------|----------------------------------|----------------------------------------------------------------|
| 空/非空      | 无输入参数                       | 未覆盖（无参查询接口）                                         |
| 枚举选项     | 布尔返回值 True/False            | 已覆盖（test_return_type_and_value，NPU 上为 False）           |
| 参数类型     | 传入非法位置/关键字参数          | 已覆盖（test_unexpected_arguments_raise）                      |
| 传参与不传参 | 无参调用与多余参数调用           | 已覆盖（test_return_type_and_value、test_unexpected_arguments_raise） |
| 等价类/边界  | 多次重复调用幂等性               | 已覆盖（test_return_type_and_value）                           |
| 正常传参场景 | 默认无参查询 NVSHMEM 可用性      | 已覆盖（test_return_type_and_value）                           |
| 异常传参场景 | 多余位置参数或关键字参数报错     | 已覆盖（test_unexpected_arguments_raise）                      |
| 混合设备类型 | 不涉及 Tensor 输入               | 未覆盖（无参全局能力查询接口）                                 |

未覆盖项及原因：
- 空/非空：接口无任何入参。
- 混合设备类型：接口不接收 Tensor 参数。
- 多卡 HCCL：本接口为纯本地构建/运行时能力查询，不涉及跨进程通信。

注意：本接口返回布尔值，无浮点 Tensor 输出；
     在 NPU 上断言具体取值 False（实现误报 True 时用例失败）。
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
class TestIsNvshmemAvailable(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_return_type_and_value(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        available = symm_mem.is_nvshmem_available()
        self.assertIsInstance(available, bool)
        if self.device_name == "npu":
            self.assertIs(available, False)
        self.assertIs(available, symm_mem.is_nvshmem_available())

    def test_unexpected_arguments_raise(self):
        for args, kw in [((0,), {}), ((), {"device": self.device_name})]:
            with self.assertRaises(TypeError):
                symm_mem.is_nvshmem_available(*args, **kw)


if __name__ == "__main__":
    run_tests()
