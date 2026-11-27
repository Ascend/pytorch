# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.set_signal_pad_size 接口功能正确性
API 名称：torch.distributed._symmetric_memory.set_signal_pad_size
API 签名：torch.distributed._symmetric_memory.set_signal_pad_size(size: int) -> None

覆盖维度表：
| 覆盖维度     | 说明                                 | 覆盖情况                                                       |
|--------------|--------------------------------------|----------------------------------------------------------------|
| 空/非空      | 传入 None 报错，有效整数正常设置     | 已覆盖（test_set_and_restore_valid_sizes、invalid_args）       |
| 枚举选项     | 不涉及离散枚举                       | 未覆盖（参数为非负整型字节大小）                               |
| 参数类型     | int 合法；str/float/None 非法        | 已覆盖（test_set_and_restore_valid_sizes、invalid_args）       |
| 传参与不传参 | 位置/关键字传参与缺省报错            | 已覆盖（test_set_and_restore_valid_sizes、invalid_args）       |
| 等价类/边界  | 0、1、默认 9216、大尺寸 1MB、负数    | 已覆盖（test_set_and_restore_valid_sizes、invalid_args）       |
| 正常传参场景 | 位置与关键字设置多种尺寸并回读验证   | 已覆盖（test_set_and_restore_valid_sizes）                     |
| 异常传参场景 | 缺参、非法类型、负数、未知关键字报错 | 已覆盖（test_invalid_arguments_raise）                         |
| 混合设备类型 | 不涉及 Tensor 输入                   | 未覆盖（全局配置设置接口）                                     |

未覆盖项及原因：
- 枚举选项：size 为连续整型字节数，无离散枚举。
- 混合设备类型：接口不接收 Tensor 参数。
- 多卡 HCCL：本接口为本地全局配置设置，不涉及跨进程通信。

注意：本接口返回 None，无浮点 Tensor 输出；
     每次设置均通过 get_signal_pad_size 断言实际生效，且在 finally 中恢复原值。
"""
import contextlib
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
class TestSetSignalPadSize(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    @contextlib.contextmanager
    def _preserve_size(self):
        orig = symm_mem.get_signal_pad_size()
        try:
            yield orig
        finally:
            symm_mem.set_signal_pad_size(orig)
        self.assertEqual(symm_mem.get_signal_pad_size(), orig)

    def test_set_and_restore_valid_sizes(self):
        self.assertIn(self.device_name, ("npu", "cuda"))
        with self._preserve_size() as orig:
            for target_size in (1, 1024, 9216, 65536, 1024 * 1024):
                self.assertIsNone(symm_mem.set_signal_pad_size(target_size))
                self.assertEqual(symm_mem.get_signal_pad_size(), target_size)
            self.assertIsNone(symm_mem.set_signal_pad_size(size=16384))
            self.assertEqual(symm_mem.get_signal_pad_size(), 16384)
            self.assertIsNone(symm_mem.set_signal_pad_size(0))
            self.assertEqual(symm_mem.get_signal_pad_size(), orig)

    def test_invalid_arguments_raise(self):
        cases = [
            (TypeError, (), {}),
            (TypeError, (None,), {}),
            (TypeError, ("4096",), {}),
            (TypeError, (1024.5,), {}),
            ((TypeError, ValueError, OverflowError), (-1,), {}),
            (TypeError, (), {"unknown": 4096}),
        ]
        with self._preserve_size():
            for exc, args, kw in cases:
                with self.assertRaises(exc):
                    symm_mem.set_signal_pad_size(*args, **kw)


if __name__ == "__main__":
    run_tests()
