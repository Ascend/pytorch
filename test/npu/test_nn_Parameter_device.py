# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.nn.Parameter.device 属性功能正确性
API 名称：torch.nn.Parameter.device
API 签名：device: torch.device (attribute, inherited from Tensor)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | N/A（属性访问）                                              | N/A         |
| 枚举选项         | CPU / NPU 设备                                               | 已覆盖      |
| 参数类型         | N/A                                                          | N/A         |
| 传参与不传参     | N/A                                                          | N/A         |
| 等价类/边界值    | 不同 dtype / shape 的 Parameter                              | 已覆盖      |
| 正常传参场景     | 访问 device 属性返回 torch.device                            | 已覆盖      |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，属性访问无异常 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回 device 类型符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch.nn as nn
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestParameterDevice(TestCase):
    """Test cases for torch.nn.Parameter.device."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_parameter_device_npu(self):
        """NPU Parameter has NPU device."""
        param = nn.Parameter(torch.randn(3, 3, device=self.device))
        self.assertEqual(param.device.type, self.device_name)

    def test_parameter_device_cpu(self):
        """CPU Parameter has CPU device."""
        param = nn.Parameter(torch.randn(3, 3))
        self.assertEqual(param.device.type, 'cpu')

    def test_parameter_device_dtype_variants(self):
        """Parameter device is correct for different dtypes on NPU."""
        for dtype in [torch.float32, torch.float16]:
            param = nn.Parameter(torch.randn(2, 2, dtype=dtype, device=self.device))
            self.assertEqual(param.device.type, self.device_name)

    def test_parameter_device_scalar(self):
        """Scalar Parameter device on NPU."""
        param = nn.Parameter(torch.tensor(5.0, device=self.device))
        self.assertEqual(param.device.type, self.device_name)


if __name__ == "__main__":
    run_tests()
