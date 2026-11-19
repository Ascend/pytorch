# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.set_default_tensor_type 接口功能正确性
API 名称：torch.set_default_tensor_type
API 签名：torch.set_default_tensor_type(t) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | t 为浮点 Tensor 类型或字符串 | 未覆盖（不适用） |
| 枚举选项 | Double/Float 两种默认类型 | 已覆盖 |
| 参数类型 | 接受 Tensor 类型（类）入参 | 已覆盖 |
| 传参与不传参 | t 为必填参数 | 已覆盖 |
| 等价类/边界值 | 切换后 torch.tensor 默认 dtype 随之改变 | 已覆盖 |
| 正常传参场景 | 设置成功且返回 None | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收类型对象、不接收 Tensor，空值与混合设备维度不适用。
- 异常传参场景：该 API 为已弃用接口，非法入参的异常类型在不同版本间不稳定，
  故不构造非法入参断言（仅在合理类型上验证语义）。
- 本 API 仅修改全局默认类型，不产生浮点数值输出，故不做跨设备数值对照；
  默认 dtype/device 在 tearDown 中还原。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testset_default_tensor_type(TestCase):
    """Test cases for torch.set_default_tensor_type (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._dtype = torch.get_default_dtype()
        self._device = torch.get_default_device()

    def tearDown(self):
        try:
            torch.set_default_dtype(self._dtype)
            torch.set_default_device(self._device)
        finally:
            super().tearDown()

    def test_returns_none(self):
        """The setter returns None."""
        self.assertIsNone(torch.set_default_tensor_type(torch.FloatTensor))

    def test_double_tensor_changes_default_dtype(self):
        """Switching to DoubleTensor makes torch.tensor default to float64."""
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.assertEqual(torch.get_default_dtype(), torch.float64)
        self.assertEqual(torch.tensor([1.2]).dtype, torch.float64)

    def test_float_tensor_restores_float32(self):
        """Switching back to FloatTensor restores the float32 default."""
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_tensor_type(torch.FloatTensor)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        self.assertEqual(torch.tensor([1.2]).dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
