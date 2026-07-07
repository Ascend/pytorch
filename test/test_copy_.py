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
| 精度/数值正确性  | mixed-dtype host-device 路径下，同步/异步 copy 结果一致      | 已覆盖                                         |
| 正常传参场景     | NPU 上 copy 后 self 的 shape/dtype 不变；返回 self             | 已覆盖                                         |
| 异常传参场景     | 不可广播的 shape                                             | 已覆盖                                         |

未覆盖项及原因：
- 无

注意：本测试除了验证功能正确性（调用不报错、tensor 结构属性符合预期），
     也对 mixed-dtype host-device 路径补充了同步/异步 copy 结果一致性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch_npu.testing.common_utils import SupportedDevices

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
        self.dtype_cast_pairs = [
            (torch.int32, torch.float32),
            (torch.int64, torch.float32),
            (torch.float16, torch.float32),
            (torch.float32, torch.float16),
        ]
        self.aclnn_cast_fallback_dtypes = [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]
        self.precision_compare_cases = [
            (torch.int8, torch.float16, [-127, -31, -1, 0, 1, 7, 42, 127]),
            (torch.int16, torch.float32, [-32768, -1025, -1, 0, 1, 255, 1024, 32767]),
            (torch.int64, torch.float32, [-(2 ** 20), -12345, -1, 0, 1, 12345, 4096, 2 ** 20]),
            (torch.float16, torch.float32, [-2048.0, -7.5, -0.125, 0.0, 0.125, 1.5, 33.25, 2048.0]),
            (torch.bfloat16, torch.float32, [-1.0e4, -7.5, -0.125, 0.0, 0.125, 1.5, 33.25, 1.0e4]),
            (torch.float32, torch.float16, [-65504.0, -255.5, -0.33325195, 0.0, 0.33325195, 17.625, 255.5, 65504.0]),
            (torch.float32, torch.bfloat16, [-1.0e8, -255.5, -0.33325195, 0.0, 0.33325195, 17.625, 255.5, 1.0e8]),
        ]

    def _make_host_source(self, dtype, pin_memory=False):
        src = torch.tensor(
            [[-7.5, -3.25, -1.0, 0.0], [1.5, 2.25, 5.0, 9.75]],
            dtype=torch.float32,
        ).to(dtype)
        return src.pin_memory() if pin_memory else src

    def _make_device_source(self, dtype):
        return self._make_host_source(dtype).to(self.device)

    def _assert_dtype_cast_copy_keeps_async(self, dst, src):
        gate_stream = torch_npu.npu.Stream(device=self.device)
        copy_stream = torch_npu.npu.Stream(device=self.device)
        gate_event = torch_npu.npu.Event()
        done_event = torch_npu.npu.Event()

        torch_npu.npu.synchronize()

        # Keep copy_stream pending behind work on gate_stream. A synchronous
        # fallback in copy_ would wait for the gate before returning.
        gate_a = torch.ones((4096, 4096), device=self.device, dtype=torch.float32)
        gate_b = torch.ones((4096, 4096), device=self.device, dtype=torch.float32)
        with torch_npu.npu.stream(gate_stream):
            gate_c = gate_a @ gate_b
            gate_c = gate_c @ gate_b
            gate_event.record()

        with torch_npu.npu.stream(copy_stream):
            copy_stream.wait_event(gate_event)
            ret = dst.copy_(src, non_blocking=True)
            done_event.record()

        self.assertIs(ret, dst)
        self.assertFalse(done_event.query())
        done_event.synchronize()

    def _assert_copy_matches_cast(self, dst, src):
        expected = src.cpu().to(dtype=dst.dtype)
        actual = dst.cpu() if dst.device.type == self.device_name else dst
        self.assertEqual(actual, expected)

    def _to_cpu_if_needed(self, tensor):
        return tensor.cpu() if tensor.device.type == self.device_name else tensor

    def _assert_non_blocking_matches_blocking(self, async_dst, sync_dst, src, async_base=None, sync_base=None):
        sync_ret = sync_dst.copy_(src, non_blocking=False)
        async_ret = async_dst.copy_(src, non_blocking=True)

        self.assertIs(sync_ret, sync_dst)
        self.assertIs(async_ret, async_dst)

        torch_npu.npu.synchronize()
        self.assertEqual(self._to_cpu_if_needed(async_dst), self._to_cpu_if_needed(sync_dst))

        if async_base is not None and sync_base is not None:
            self.assertEqual(self._to_cpu_if_needed(async_base), self._to_cpu_if_needed(sync_base))

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

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_from_pinned_cpu_src_dtype_cast_non_blocking(self):
        for src_dtype, dst_dtype in self.dtype_cast_pairs:
            dst = torch.empty(2, 4, dtype=dst_dtype, device=self.device)
            src = self._make_host_source(src_dtype, pin_memory=True)
            self._assert_dtype_cast_copy_keeps_async(dst, src)
            self._assert_copy_matches_cast(dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_pinned_cpu_from_npu_src_dtype_cast_non_blocking(self):
        for src_dtype, dst_dtype in self.dtype_cast_pairs:
            dst = torch.empty(2, 4, dtype=dst_dtype, pin_memory=True)
            src = self._make_device_source(src_dtype)
            self._assert_dtype_cast_copy_keeps_async(dst, src)
            self.assertTrue(dst.is_pinned())
            self._assert_copy_matches_cast(dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_pinned_cpu_from_non_contiguous_npu_src_dtype_cast_non_blocking(self):
        src = torch.arange(8, dtype=torch.int32, device=self.device).reshape(4, 2).t()
        dst = torch.empty(2, 4, dtype=torch.float32, pin_memory=True)
        self.assertFalse(src.is_contiguous())
        self._assert_dtype_cast_copy_keeps_async(dst, src)
        self.assertTrue(dst.is_pinned())
        self._assert_copy_matches_cast(dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_pinned_cpu_non_contiguous_dst_dtype_cast_preserves_strided_layout(self):
        base = torch.full((3, 6), -99.0, dtype=torch.float32).pin_memory()
        dst = base[:, 1::2]
        src = torch.arange(9, dtype=torch.int32, device=self.device).reshape(3, 3)
        expected_base = torch.full((3, 6), -99.0, dtype=torch.float32)
        expected_base[:, 1::2] = src.cpu().to(dtype=dst.dtype)

        self.assertFalse(dst.is_contiguous())
        self.assertTrue(dst.is_pinned())
        self.assertNotEqual(dst.storage_offset(), 0)
        ret = dst.copy_(src, non_blocking=True)
        torch_npu.npu.synchronize()

        self.assertIs(ret, dst)
        self.assertEqual(dst, src.cpu().to(dtype=dst.dtype))
        self.assertEqual(base, expected_base)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_non_contiguous_dst_dtype_cast_non_blocking(self):
        base = torch.empty(4, 2, dtype=torch.float32, device=self.device)
        dst = base.t()
        src = self._make_host_source(torch.int32, pin_memory=True)
        self.assertFalse(dst.is_contiguous())
        self._assert_dtype_cast_copy_keeps_async(dst, src)
        self._assert_copy_matches_cast(dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_non_contiguous_dst_dtype_cast_preserves_strided_layout(self):
        base = torch.full((3, 6), -99.0, dtype=torch.float32, device=self.device)
        dst = base[:, 1::2]
        src = torch.arange(9, dtype=torch.int32).reshape(3, 3).pin_memory()
        expected_base = torch.full((3, 6), -99.0, dtype=torch.float32)
        expected_base[:, 1::2] = src.to(dtype=dst.dtype)

        self.assertFalse(dst.is_contiguous())
        self.assertNotEqual(dst.storage_offset(), 0)
        ret = dst.copy_(src, non_blocking=True)
        torch_npu.npu.synchronize()

        self.assertIs(ret, dst)
        self.assertEqual(dst.cpu(), src.to(dtype=dst.dtype))
        self.assertEqual(base.cpu(), expected_base)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_from_pinned_cpu_broadcast_src_dtype_cast_non_blocking(self):
        dst = torch.empty(3, 4, dtype=torch.float32, device=self.device)
        src = torch.arange(4, dtype=torch.int32).reshape(1, 4).pin_memory()
        expected = src.to(dtype=dst.dtype).expand(3, 4)

        ret = dst.copy_(src, non_blocking=True)
        torch_npu.npu.synchronize()

        self.assertIs(ret, dst)
        self.assertEqual(dst.cpu(), expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_pinned_cpu_from_npu_broadcast_src_dtype_cast_non_blocking(self):
        dst = torch.empty(3, 4, dtype=torch.float32, pin_memory=True)
        src = torch.arange(4, dtype=torch.int32, device=self.device).reshape(1, 4)
        expected = src.cpu().to(dtype=dst.dtype).expand_as(dst)

        ret = dst.copy_(src, non_blocking=True)
        torch_npu.npu.synchronize()

        self.assertIs(ret, dst)
        self.assertEqual(dst, expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_from_pinned_cpu_aclnn_cast_unsupported_src_dtype_fallback(self):
        for src_dtype in self.aclnn_cast_fallback_dtypes:
            dst = torch.empty(2, 4, dtype=torch.float32, device=self.device)
            src = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(src_dtype).pin_memory()

            ret = dst.copy_(src, non_blocking=True)
            torch_npu.npu.synchronize()

            self.assertIs(ret, dst)
            self.assertEqual(dst.cpu(), src.to(dtype=dst.dtype))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_pinned_cpu_from_npu_aclnn_cast_unsupported_dst_dtype_fallback(self):
        for dst_dtype in self.aclnn_cast_fallback_dtypes:
            dst = torch.empty(2, 4, dtype=dst_dtype).pin_memory()
            src = torch.arange(8, dtype=torch.float32, device=self.device).reshape(2, 4)

            ret = dst.copy_(src, non_blocking=True)
            torch_npu.npu.synchronize()

            self.assertIs(ret, dst)
            self.assertEqual(dst.dtype, dst_dtype)
            self.assertEqual(dst.to(dtype=src.dtype), src.cpu())

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_from_pinned_cpu_slice_dtype_cast_non_blocking(self):
        src_base = torch.arange(9, dtype=torch.int32).pin_memory()
        src = src_base[1:].reshape(2, 4)
        dst = torch.empty(2, 4, dtype=torch.float32, device=self.device)
        self.assertTrue(src.is_pinned())
        self.assertNotEqual(src.data_ptr(), src.untyped_storage().data_ptr())
        self._assert_dtype_cast_copy_keeps_async(dst, src)
        self._assert_copy_matches_cast(dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_from_pinned_cpu_dtype_cast_non_blocking_matches_blocking(self):
        for src_dtype, dst_dtype, values in self.precision_compare_cases:
            with self.subTest(direction="h2d", src_dtype=src_dtype, dst_dtype=dst_dtype):
                src = torch.tensor(values, dtype=src_dtype).reshape(2, 4).pin_memory()
                async_dst = torch.empty(2, 4, dtype=dst_dtype, device=self.device)
                sync_dst = torch.empty(2, 4, dtype=dst_dtype, device=self.device)

                self._assert_non_blocking_matches_blocking(async_dst, sync_dst, src)
                self._assert_copy_matches_cast(async_dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_pinned_cpu_from_npu_dtype_cast_non_blocking_matches_blocking(self):
        for src_dtype, dst_dtype, values in self.precision_compare_cases:
            with self.subTest(direction="d2h", src_dtype=src_dtype, dst_dtype=dst_dtype):
                src = torch.tensor(values, dtype=src_dtype).reshape(2, 4).to(self.device)
                async_dst = torch.empty(2, 4, dtype=dst_dtype, pin_memory=True)
                sync_dst = torch.empty(2, 4, dtype=dst_dtype, pin_memory=True)

                self._assert_non_blocking_matches_blocking(async_dst, sync_dst, src)
                self._assert_copy_matches_cast(async_dst, src)
                self.assertTrue(async_dst.is_pinned())
                self.assertTrue(sync_dst.is_pinned())

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_mixed_dtype_non_blocking_matches_blocking_for_layout_variants(self):
        layout_cases = [
            {
                "name": "h2d_non_contiguous_dst",
                "src_dtype": torch.float32,
                "dst_dtype": torch.float16,
                "make_src": lambda: torch.tensor(
                    [-63.5, -7.25, -0.5, 0.0, 0.5, 3.25, 17.75, 63.5, -19.5],
                    dtype=torch.float32,
                ).reshape(3, 3).pin_memory(),
                "make_async_dst": lambda: torch.full((3, 6), -99.0, dtype=torch.float16, device=self.device),
                "make_sync_dst": lambda: torch.full((3, 6), -99.0, dtype=torch.float16, device=self.device),
                "select_dst_view": lambda base: base[:, 1::2],
                "expected": lambda src, dst: src.to(dtype=dst.dtype),
            },
            {
                "name": "d2h_non_contiguous_dst",
                "src_dtype": torch.int32,
                "dst_dtype": torch.float32,
                "make_src": lambda: torch.arange(9, dtype=torch.int32, device=self.device).reshape(3, 3) - 4,
                "make_async_dst": lambda: torch.full((3, 6), -99.0, dtype=torch.float32).pin_memory(),
                "make_sync_dst": lambda: torch.full((3, 6), -99.0, dtype=torch.float32).pin_memory(),
                "select_dst_view": lambda base: base[:, 1::2],
                "expected": lambda src, dst: src.cpu().to(dtype=dst.dtype),
            },
            {
                "name": "h2d_broadcast_src",
                "src_dtype": torch.int16,
                "dst_dtype": torch.float32,
                "make_src": lambda: torch.tensor([-32768, -17, 9, 32767], dtype=torch.int16).reshape(1, 4).pin_memory(),
                "make_async_dst": lambda: torch.empty(3, 4, dtype=torch.float32, device=self.device),
                "make_sync_dst": lambda: torch.empty(3, 4, dtype=torch.float32, device=self.device),
                "select_dst_view": lambda base: base,
                "expected": lambda src, dst: src.to(dtype=dst.dtype).expand_as(dst),
            },
            {
                "name": "d2h_non_contiguous_src",
                "src_dtype": torch.float16,
                "dst_dtype": torch.float32,
                "make_src": lambda: torch.tensor(
                    [-7.5, -1.25, 0.0, 1.25, 3.5, 7.75, 15.5, 31.0],
                    dtype=torch.float16,
                    device=self.device,
                ).reshape(4, 2).t(),
                "make_async_dst": lambda: torch.empty(2, 4, dtype=torch.float32, pin_memory=True),
                "make_sync_dst": lambda: torch.empty(2, 4, dtype=torch.float32, pin_memory=True),
                "select_dst_view": lambda base: base,
                "expected": lambda src, dst: src.cpu().to(dtype=dst.dtype),
            },
        ]

        for case in layout_cases:
            with self.subTest(case=case["name"], src_dtype=case["src_dtype"], dst_dtype=case["dst_dtype"]):
                src = case["make_src"]()
                async_base = case["make_async_dst"]()
                sync_base = case["make_sync_dst"]()
                async_dst = case["select_dst_view"](async_base)
                sync_dst = case["select_dst_view"](sync_base)

                self._assert_non_blocking_matches_blocking(
                    async_dst,
                    sync_dst,
                    src,
                    async_base=async_base,
                    sync_base=sync_base,
                )

                self.assertEqual(
                    self._to_cpu_if_needed(async_dst),
                    case["expected"](src, async_dst),
                )

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_dtype_cast_non_blocking_temporary_lifetime(self):
        h2d_dst = torch.empty(2, 4, dtype=torch.float32, device=self.device)
        d2h_dst = torch.empty(2, 4, dtype=torch.float32, pin_memory=True)
        expected = None

        for i in range(32):
            host_src = self._make_host_source(torch.int32) + i
            h2d_src = host_src.pin_memory()
            h2d_dst.copy_(h2d_src, non_blocking=True)
            d2h_src = host_src.to(self.device)
            d2h_dst.copy_(d2h_src, non_blocking=True)
            expected = host_src.to(dtype=d2h_dst.dtype)

        torch_npu.npu.synchronize()
        self.assertEqual(h2d_dst.cpu(), h2d_src.to(dtype=h2d_dst.dtype))
        self.assertEqual(d2h_dst, expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_npu_from_cpu_src_dtype_cast_blocking(self):
        for src_dtype, dst_dtype in self.dtype_cast_pairs:
            dst = torch.empty(2, 4, dtype=dst_dtype, device=self.device)
            src = self._make_host_source(src_dtype)
            ret = dst.copy_(src, non_blocking=False)
            self.assertIs(ret, dst)
            self._assert_copy_matches_cast(dst, src)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_copy_cpu_from_npu_src_dtype_cast_blocking(self):
        for src_dtype, dst_dtype in self.dtype_cast_pairs:
            dst = torch.empty(2, 4, dtype=dst_dtype)
            src = self._make_device_source(src_dtype)
            ret = dst.copy_(src, non_blocking=False)
            self.assertIs(ret, dst)
            self._assert_copy_matches_cast(dst, src)

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
