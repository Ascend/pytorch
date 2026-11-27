# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.get_signal_pad_size 接口功能正确性
API 名称：torch.distributed._symmetric_memory.get_signal_pad_size
API 签名：torch.distributed._symmetric_memory.get_signal_pad_size() -> int

覆盖维度表：
| 覆盖维度     | 说明                                 | 覆盖情况                                                       |
|--------------|--------------------------------------|----------------------------------------------------------------|
| 空/非空      | 无输入参数                           | 未覆盖（无参查询接口）                                         |
| 枚举选项     | 不涉及枚举参数                       | 未覆盖（无参查询接口）                                         |
| 参数类型     | 传入非法位置/关键字参数              | 已覆盖（test_unexpected_arguments_raise）                      |
| 传参与不传参 | 无参调用与多余参数调用               | 已覆盖（test_reflects_set_signal_pad_size、test_unexpected_arguments_raise） |
| 等价类/边界  | 默认值 9216、修改后同步更新与 0 边界 | 已覆盖（test_reflects_set_signal_pad_size）                    |
| 正常传参场景 | 默认查询、修改后查询与幂等查询       | 已覆盖（test_reflects_set_signal_pad_size）                    |
| 异常传参场景 | 多余位置参数或关键字参数报错         | 已覆盖（test_unexpected_arguments_raise）                      |
| 混合设备类型 | 不涉及 Tensor 输入                   | 未覆盖（全局配置查询接口）                                     |

未覆盖项及原因：
- 空/非空与枚举选项：接口无任何入参。
- 混合设备类型：接口不接收 Tensor 参数。
- 多卡 HCCL：本接口为本地全局配置查询，不涉及跨进程通信。

注意：本接口返回整型字节数，无浮点 Tensor 输出；
     验证默认值及与 set_signal_pad_size 的状态联动，且在 finally 中恢复原值。
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
class TestGetSignalPadSize(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_reflects_set_signal_pad_size(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        original_size = symm_mem.get_signal_pad_size()
        self.assertIsInstance(original_size, int)
        self.assertNotIsInstance(original_size, bool)
        self.assertEqual(original_size, 9216)
        self.assertEqual(symm_mem.get_signal_pad_size(), original_size)
        try:
            for target_size in (4096, 65536, 1024 * 1024):
                symm_mem.set_signal_pad_size(target_size)
                self.assertEqual(symm_mem.get_signal_pad_size(), target_size)
            symm_mem.set_signal_pad_size(0)
            self.assertEqual(symm_mem.get_signal_pad_size(), original_size)
        finally:
            symm_mem.set_signal_pad_size(original_size)
        self.assertEqual(symm_mem.get_signal_pad_size(), original_size)

    def test_unexpected_arguments_raise(self):
        for args, kw in [((9216,), {}), ((), {"size": 9216})]:
            with self.assertRaises(TypeError):
                symm_mem.get_signal_pad_size(*args, **kw)


if __name__ == "__main__":
    run_tests()
