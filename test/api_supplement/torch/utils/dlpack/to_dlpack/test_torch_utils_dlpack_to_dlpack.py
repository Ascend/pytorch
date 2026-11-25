# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.dlpack.to_dlpack 接口功能正确性
API 名称：torch.utils.dlpack.to_dlpack
API 签名：torch.utils.dlpack.to_dlpack(tensor)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空 Tensor 与非空浮点/整数 Tensor           | 已覆盖（empty_tensor、cpu_and_device_export）          |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | float32 与 int64 Tensor（CPU 及目标设备）  | 已覆盖（cpu_and_device_export、int64_dtype_export）    |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（仅接收单个必填 tensor 参数，无可选默认参数）   |
| 等价类/边界值    | 零元素 Tensor 导出与共享内存原地可见性     | 已覆盖（empty_tensor、shared_memory_export）           |
| 正常传参场景     | 导出 PyCapsule 并在 CPU/目标设备往返还原   | 已覆盖（cpu_and_device_export、int64_dtype_export、shared_memory_export） |
| 异常传参场景     | 非 Tensor 对象与缺失必填参数               | 已覆盖（non_tensor_raises、missing_arg_raises）        |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（单 Tensor 导出接口，无多 Tensor 混合输入）     |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：仅接收单个必填 tensor 参数，无可选默认参数。
- 混合设备类型：单 Tensor 导出接口，不涉及多 Tensor 混合设备输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestToDlpack(TestCase):
    """Test cases for torch.utils.dlpack.to_dlpack."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_cpu_and_device_export(self):
        """to_dlpack returns a PyCapsule that roundtrips accurately on CPU and device."""
        cpu_t = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)
        cpu_cap = to_dlpack(cpu_t)
        self.assertEqual(type(cpu_cap).__name__, "PyCapsule")
        cpu_back = from_dlpack(cpu_cap)
        self.assertEqual(cpu_back.shape, cpu_t.shape)
        self.assertEqual(cpu_back.dtype, cpu_t.dtype)
        torch.testing.assert_close(cpu_back, cpu_t)

        dev_t = cpu_t.to(self.device)
        dev_cap = to_dlpack(dev_t)
        self.assertEqual(type(dev_cap).__name__, "PyCapsule")
        dev_back = from_dlpack(dev_cap)
        self.assertEqual(dev_back.shape, dev_t.shape)
        self.assertEqual(dev_back.dtype, dev_t.dtype)
        self.assertEqual(dev_back.device.type, self.device.type)
        torch.testing.assert_close(dev_back.cpu(), cpu_back)

    def test_int64_dtype_export(self):
        """Exporting an int64 tensor preserves exact integer values and dtype."""
        t = torch.tensor([10, 20, -30], dtype=torch.int64, device=self.device)
        cap = to_dlpack(t)
        self.assertEqual(type(cap).__name__, "PyCapsule")
        restored = from_dlpack(cap)
        self.assertEqual(restored.dtype, torch.int64)
        self.assertEqual(restored.cpu().tolist(), [10, 20, -30])

    def test_empty_tensor(self):
        """Exporting an empty tensor produces a valid PyCapsule with matching shape."""
        t = torch.empty(0, 2, dtype=torch.float32, device=self.device)
        cap = to_dlpack(t)
        self.assertEqual(type(cap).__name__, "PyCapsule")
        restored = from_dlpack(cap)
        self.assertEqual(restored.shape, (0, 2))
        self.assertEqual(restored.numel(), 0)

    def test_shared_memory_export(self):
        """Mutating source tensor is reflected in tensor restored from its capsule."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
        restored = from_dlpack(to_dlpack(t))
        t[1] = 99.0
        torch.testing.assert_close(restored.cpu(), torch.tensor([1.0, 99.0, 3.0]))

    def test_non_tensor_raises(self):
        """Passing a non-tensor object to to_dlpack raises TypeError."""
        with self.assertRaises(TypeError):
            to_dlpack([1, 2, 3])

    def test_missing_arg_raises(self):
        """Calling to_dlpack without tensor raises TypeError."""
        with self.assertRaises(TypeError):
            to_dlpack()


if __name__ == "__main__":
    run_tests()
