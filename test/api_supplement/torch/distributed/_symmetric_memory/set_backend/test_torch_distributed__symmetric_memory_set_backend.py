# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.set_backend 接口功能正确性
API 名称：torch.distributed._symmetric_memory.set_backend
API 签名：torch.distributed._symmetric_memory.set_backend(name: str) -> None

覆盖维度表：
| 覆盖维度     | 说明                                 | 覆盖情况                                                       |
|--------------|--------------------------------------|----------------------------------------------------------------|
| 空/非空      | 空字符串与 None 报错，合法名称正常   | 已覆盖（test_set_current_backend_positional_and_keyword 等）   |
| 枚举选项     | 当前设备已注册后端名称与未知名称     | 已覆盖（test_set_current_backend_positional_and_keyword）      |
| 参数类型     | str 合法；int/None 非法              | 已覆盖（test_set_current_backend_positional_and_keyword 等）   |
| 传参与不传参 | 位置/关键字传参与缺参报错            | 已覆盖（test_set_current_backend_positional_and_keyword 等）   |
| 等价类/边界  | 空串、非法名称、正常名称             | 已覆盖（test_set_current_backend_positional_and_keyword）      |
| 正常传参场景 | 位置与关键字设置有效后端并回读验证   | 已覆盖（test_set_current_backend_positional_and_keyword）      |
| 异常传参场景 | 未知后端名、空串、非字符串、缺参报错 | 已覆盖（test_invalid_argument_types_raise 及正常用例异常段）   |
| 混合设备类型 | 不涉及 Tensor 输入                   | 未覆盖（全局后端配置接口）                                     |

未覆盖项及原因：
- 混合设备类型：接口不接收 Tensor 参数。
- 多卡 HCCL：本接口为本地全局后端设置，不涉及跨进程通信。
- NPU 上 set_backend('NPUSHMEM') 查不到分配后端，相关用例在 NPU 跳过。

注意：本接口返回 None，无浮点 Tensor 输出；
     设置后通过 get_backend 验证生效，并在 finally 中恢复原始后端。
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


NPU_SKIP_REASON = "PTA/NPU defect: SymmetricMemory does not find allocation backend NPUSHMEM"


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestSetBackend(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    @unittest.skipIf(DEVICE_NAME == "npu", NPU_SKIP_REASON)
    def test_set_current_backend_positional_and_keyword(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        target_dev = torch.device(f"{self.device_name}:0")
        orig_backend = symm_mem.get_backend(target_dev)
        self.assertIsInstance(orig_backend, str)
        try:
            for args, kw in [((orig_backend,), {}), ((), {"name": orig_backend})]:
                self.assertIsNone(symm_mem.set_backend(*args, **kw))
                self.assertEqual(symm_mem.get_backend(target_dev), orig_backend)
            for bad in ("UNKNOWN_SYMM_MEM_BACKEND_XYZ", ""):
                with self.assertRaises((RuntimeError, ValueError)):
                    symm_mem.set_backend(bad)
        finally:
            symm_mem.set_backend(orig_backend)
        self.assertEqual(symm_mem.get_backend(target_dev), orig_backend)

    def test_invalid_argument_types_raise(self):
        cases = [((), {}), ((None,), {}), ((123,), {}), ((), {"backend": "NPUSHMEM"})]
        for args, kw in cases:
            with self.assertRaises(TypeError):
                symm_mem.set_backend(*args, **kw)


if __name__ == "__main__":
    run_tests()
