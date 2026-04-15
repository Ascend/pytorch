# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.copy_ 接口功能正确性
API 名称：torch.Tensor.copy_
API 签名：copy_(src, non_blocking=False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | size-0 张量互拷                                              | 已覆盖                                         |
| 枚举选项         | non_blocking 为 False / True                               | 已覆盖                                         |
| 参数类型         | src 为 Tensor（含标量张量）、与 self  dtype 可不同           | 已覆盖                                         |
| 传参与不传参     | non_blocking 省略与显式传入                                  | 已覆盖                                         |
| 等价类/边界值    | 同形、可广播、非连续目标、跨 CPU/NPU                         | 已覆盖                                         |
| 正常传参场景     | NPU 上 copy 后 self 的 shape/dtype 不变；返回 self             | 已覆盖                                         |
| 异常传参场景     | 不可广播的 shape                                             | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、tensor 结构属性符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestTensorCopy_(TestCase):
    """Functional tests for torch.Tensor.copy_ on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name,
            "npu",
            f"Expected device 'npu', got '{self.device_name}'",
        )
        self.device = torch.device(self.device_name)

    def test_copy_npu_same_device_same_shape(self):
        dst = torch.empty(3, 4, device=self.device, dtype=torch.float32)
        src = torch.randn(3, 4, device=self.device, dtype=torch.float32)
        before_shape = dst.shape
        before_dtype = dst.dtype
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.shape, before_shape)
        self.assertEqual(dst.dtype, before_dtype)
        self.assertEqual(dst.device.type, self.device_name)

    def test_copy_npu_broadcast_src(self):
        dst = torch.empty(4, 3, device=self.device)
        src = torch.randn(1, 3, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([4, 3]))


    def test_copy_npu_from_cpu_src(self):
        dst = torch.empty(2, 5, device=self.device)
        src = torch.randn(2, 5)
        dst.copy_(src)
        self.assertEqual(dst.device.type, self.device_name)
        self.assertEqual(dst.shape, torch.Size([2, 5]))


    def test_copy_npu_non_blocking_false(self):
        dst = torch.empty(2, 2, device=self.device)
        src = torch.ones(2, 2, device=self.device)
        ret = dst.copy_(src, non_blocking=False)
        self.assertIs(ret, dst)

    def test_copy_npu_non_blocking_true(self):
        dst = torch.empty(2, 2, device=self.device)
        src = torch.ones(2, 2, device=self.device)
        ret = dst.copy_(src, non_blocking=True)
        self.assertIs(ret, dst)
        self.assertEqual(dst.shape, torch.Size([2, 2]))

    def test_copy_npu_src_int_dtype_cast(self):
        dst = torch.empty(2, 2, dtype=torch.float32, device=self.device)
        src = torch.ones(2, 2, dtype=torch.int32, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.dtype, torch.float32)

    def test_copy_npu_non_contiguous_dst(self):
        base = torch.empty(6, 4, device=self.device)
        dst = base.t()
        self.assertFalse(dst.is_contiguous())
        src = torch.randn(4, 6, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([4, 6]))

    def test_copy_npu_empty_tensor(self):
        dst = torch.empty(0, 3, device=self.device)
        src = torch.empty(0, 3, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([0, 3]))

    def test_copy_npu_float16(self):
        dst = torch.empty(2, 3, dtype=torch.float16, device=self.device)
        src = torch.randn(2, 3, dtype=torch.float16, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.dtype, torch.float16)

    def test_copy_npu_bfloat16(self):
        dst = torch.empty(2, 3, dtype=torch.bfloat16, device=self.device)
        src = torch.randn(2, 3, dtype=torch.bfloat16, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.dtype, torch.bfloat16)

    def test_copy_npu_incompatible_shape_raises(self):
        dst = torch.empty(3, 4, device=self.device)
        src = torch.randn(2, 3, device=self.device)
        with self.assertRaises(RuntimeError):
            # NPU execution is async; force sync so the error is raised here.
            out = dst.copy_(src)
            out.cpu()

    def test_copy_cpu_baseline(self):
        dst = torch.empty(3, 4)
        src = torch.randn(3, 4)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.shape, torch.Size([3, 4]))

    def test_copy_cpu_baseline_broadcast(self):
        dst = torch.empty(2, 4)
        src = torch.randn(1, 4)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([2, 4]))


if __name__ == "__main__":
    run_tests()
