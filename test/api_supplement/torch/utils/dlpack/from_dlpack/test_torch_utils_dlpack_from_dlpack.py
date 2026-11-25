# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.dlpack.from_dlpack 接口功能正确性
API 名称：torch.utils.dlpack.from_dlpack
API 签名：torch.utils.dlpack.from_dlpack(ext_tensor, *, device=None, copy=None)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空与非空 Tensor 往返 | 已覆盖（int_dtype_and_empty_tensor、capsule_roundtrip_cpu_and_device） |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | PyCapsule/__dlpack__/float32/int64         | 已覆盖（direct_protocol、int_dtype_and_empty_tensor）  |
| 传参与不传参     | device/copy 可选关键字参数                 | 未覆盖（当前用例覆盖默认 device/copy=None 路径）       |
| 等价类/边界值    | 共享底层存储语义与零元素 Tensor 往返       | 已覆盖（shared_memory_semantics、int_dtype_and_empty_tensor） |
| 正常传参场景     | CPU/目标设备 PyCapsule 与协议导入          | 已覆盖（capsule_roundtrip_cpu_and_device、direct_protocol） |
| 异常传参场景     | 非法非 DLPack 对象与缺失必填参数           | 已覆盖（invalid_input_raises、missing_arg_raises）     |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（单外部对象导入接口，无多 Tensor 混合输入）     |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：当前单测聚焦默认 device/copy=None 的零拷贝往返语义，未展开显式关键字参数组合。
- 混合设备类型：单输入导入接口，不涉及多 Tensor 混合设备输入。

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
class TestFromDlpack(TestCase):
    """Test cases for torch.utils.dlpack.from_dlpack."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_capsule_roundtrip_cpu_and_device(self):
        """to_dlpack -> from_dlpack roundtrip preserves values, shape, dtype, device."""
        cpu_src = torch.tensor([[1.25, -2.5], [3.75, 4.0]], dtype=torch.float32)
        cpu_out = from_dlpack(to_dlpack(cpu_src))
        self.assertEqual(cpu_out.shape, cpu_src.shape)
        self.assertEqual(cpu_out.dtype, cpu_src.dtype)
        self.assertEqual(cpu_out.device.type, "cpu")
        torch.testing.assert_close(cpu_out, cpu_src)

        dev_src = cpu_src.to(self.device)
        dev_out = from_dlpack(to_dlpack(dev_src))
        self.assertEqual(dev_out.shape, dev_src.shape)
        self.assertEqual(dev_out.dtype, dev_src.dtype)
        self.assertEqual(dev_out.device.type, self.device.type)
        torch.testing.assert_close(dev_out.cpu(), cpu_out)

    def test_direct_protocol(self):
        """Passing a tensor directly uses the __dlpack__ protocol on CPU and device."""
        cpu_t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        dev_t = cpu_t.to(self.device)
        cpu_res = from_dlpack(cpu_t)
        dev_res = from_dlpack(dev_t)
        self.assertEqual(dev_res.device.type, self.device.type)
        torch.testing.assert_close(dev_res.cpu(), cpu_res)

    def test_shared_memory_semantics(self):
        """Tensor returned by from_dlpack shares underlying storage with source."""
        src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=self.device)
        imported = from_dlpack(to_dlpack(src))
        imported[0] = 42.0
        torch.testing.assert_close(src.cpu(), torch.tensor([42.0, 2.0, 3.0]))

    def test_int_dtype_and_empty_tensor(self):
        """Handles int64 tensors and zero-element tensors accurately."""
        int_t = torch.tensor([5, -3, 9], dtype=torch.int64, device=self.device)
        int_out = from_dlpack(to_dlpack(int_t))
        self.assertEqual(int_out.dtype, torch.int64)
        self.assertEqual(int_out.cpu().tolist(), [5, -3, 9])

        empty_t = torch.empty(0, 4, dtype=torch.float32, device=self.device)
        empty_out = from_dlpack(to_dlpack(empty_t))
        self.assertEqual(empty_out.shape, (0, 4))

    def test_invalid_input_raises(self):
        """Invalid non-capsule/non-DLPack input raises TypeError or RuntimeError."""
        with self.assertRaises((TypeError, RuntimeError)):
            from_dlpack(12345)

    def test_missing_arg_raises(self):
        """Calling from_dlpack without ext_tensor raises TypeError."""
        with self.assertRaises(TypeError):
            from_dlpack()


if __name__ == "__main__":
    run_tests()
