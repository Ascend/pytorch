# Copyright (c) 2026, Huawei Technologies Co., Ltd
"""
Empty_strided_npu fast path tests for NPU backend.

This test file validates the torch_npu._C._empty_strided_npu fast path
implementation, which is used by inductor for dispatcher-free NPU memory allocation.
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestEmptyStridedNpu(TestCase):
    """
    Test suite for torch_npu._C._empty_strided_npu fast path.

    This fast path is used by inductor to bypass the dispatcher overhead
    when allocating strided NPU tensors during compilation.
    """

    def test_empty_strided_npu_empty_tensor(self, device="npu"):
        """Test _empty_strided_npu with tensors containing zero dimensions."""
        # Test tensor with zero-sized dimensions
        sizes = (0, 3, 0)
        strides = (0, 0, 0)
        result = torch_npu._C._empty_strided_npu(sizes, strides, torch.float32)

        self.assertEqual(result.shape, torch.Size([0, 3, 0]))
        self.assertEqual(result.stride(), (0, 0, 0))
        self.assertEqual(result.device.type, "npu")

    def test_empty_strided_npu_various_dtypes(self, device="npu"):
        """Test _empty_strided_npu with different data types."""
        dtypes = [
            torch.float32,
            torch.float16,
            torch.int32,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]

        for dtype in dtypes:
            sizes = (2, 3)
            strides = (3, 1)
            result = torch_npu._C._empty_strided_npu(sizes, strides, dtype)

            self.assertEqual(result.shape, torch.Size([2, 3]))
            self.assertEqual(result.stride(), (3, 1))
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device.type, "npu")

    def test_empty_strided_npu_complex_strides(self, device="npu"):
        """Test _empty_strided_npu with non-contiguous stride patterns."""
        test_cases = [
            # (sizes, strides, description)
            ((4, 4), (8, 1), "non-contiguous strides"),
            ((3, 5), (10, 1), "larger stride"),
            ((2, 3, 4), (12, 4, 1), "3D contiguous"),
            ((2, 2, 2), (4, 2, 1), "3D non-contiguous"),
        ]

        for sizes, strides, desc in test_cases:
            result = torch_npu._C._empty_strided_npu(sizes, strides, torch.float32)

            self.assertEqual(result.shape, torch.Size(sizes),
                             f"Failed for {desc}: shape mismatch")
            self.assertEqual(result.stride(), strides,
                             f"Failed for {desc}: stride mismatch")
            self.assertEqual(result.device.type, "npu",
                             f"Failed for {desc}: device mismatch")

    def test_empty_strided_npu_broadcast_strides(self, device="npu"):
        """Test _empty_strided_npu with zero (broadcast) strides."""
        # Zero stride is used for broadcasting
        sizes = (3, 4)
        strides = (0, 1)  # broadcast along first dimension
        result = torch_npu._C._empty_strided_npu(sizes, strides, torch.float32)

        self.assertEqual(result.shape, torch.Size([3, 4]))
        self.assertEqual(result.stride(), (0, 1))

    def test_empty_strided_npu_storage_size(self, device="npu"):
        """Test that _empty_strided_npu allocates correct storage size."""
        # Create a tensor with non-contiguous strides
        sizes = (3, 4)
        strides = (8, 1)

        fast_result = torch_npu._C._empty_strided_npu(sizes, strides, torch.float32)
        normal_result = torch.empty_strided(sizes, strides, device=device, dtype=torch.float32)

        # Storage sizes should match
        fast_storage = fast_result.storage().size()
        normal_storage = normal_result.storage().size()

        self.assertEqual(fast_storage, normal_storage,
                         "Storage size mismatch between fast and normal path")

        # Storage should be large enough to hold the tensor
        expected_min_storage = 1 + (sizes[0] - 1) * strides[0] + (sizes[1] - 1) * strides[1]
        self.assertGreaterEqual(fast_storage, expected_min_storage,
                                "Storage size too small for the given shape and strides")

    def test_empty_strided_npu_with_new_empty_strided(self, device="npu"):
        """Test that new_empty_strided works correctly (it may use the fast path internally)."""
        x = torch.ones(()).to(device=device)
        x_new = x.new_empty_strided([2, 3], [3, 1], dtype=torch.float32)

        self.assertEqual(x_new.shape, torch.Size([2, 3]))
        self.assertEqual(x_new.stride(), (3, 1))
        self.assertEqual(x_new.device.type, device)
        self.assertEqual(x_new.dtype, torch.float32)

    @Dtypes(torch.float32, torch.float16, torch.int32)
    def test_empty_strided_npu_with_decorator(self, dtype, device="npu"):
        """Test _empty_strided_npu with @Dtypes decorator for multiple types."""
        sizes = (3, 4)
        strides = (4, 1)
        result = torch_npu._C._empty_strided_npu(sizes, strides, dtype)

        self.assertEqual(result.shape, torch.Size([3, 4]))
        self.assertEqual(result.stride(), (4, 1))
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.device.type, "npu")

    def test_empty_strided_npu_deterministic_mode_consistency(self, device="npu"):
        """
        Test that _empty_strided_npu behaves identically to torch.empty_strided
        when deterministic algorithms are enabled.

        This is critical for inductor correctness: if the fast path allocates
        memory differently than the normal path under deterministic mode,
        compiled results may differ from eager mode results.
        """
        # Save original state
        original_deterministic = torch.are_deterministic_algorithms_enabled()

        try:
            # Enable deterministic algorithms
            torch.use_deterministic_algorithms(True)

            test_cases = [
                ((2, 3), (3, 1), torch.float32, "contiguous 2D"),
                ((4, 5), (10, 1), torch.float16, "non-contiguous 2D"),
                ((3, 4, 5), (20, 5, 1), torch.float32, "3D contiguous"),
                ((0, 3), (0, 0), torch.float32, "empty tensor"),
                ((2, 2), (4, 1), torch.int32, "int32 type"),
            ]

            for sizes, strides, dtype, desc in test_cases:
                # Allocate using fast path
                fast_result = torch_npu._C._empty_strided_npu(sizes, strides, dtype)

                # Allocate using normal path
                normal_result = torch.empty_strided(sizes, strides, device=device, dtype=dtype)

                # Verify metadata matches exactly
                self.assertEqual(fast_result.shape, normal_result.shape,
                                 f"[{desc}] Shape mismatch in deterministic mode")
                self.assertEqual(fast_result.stride(), normal_result.stride(),
                                 f"[{desc}] Stride mismatch in deterministic mode")
                self.assertEqual(fast_result.dtype, normal_result.dtype,
                                 f"[{desc}] Dtype mismatch in deterministic mode")
                self.assertEqual(fast_result.device, normal_result.device,
                                 f"[{desc}] Device mismatch in deterministic mode")

                # Verify storage size matches (critical for deterministic memory usage)
                fast_storage_size = fast_result.storage().size()
                normal_storage_size = normal_result.storage().size()
                self.assertEqual(fast_storage_size, normal_storage_size,
                                 f"[{desc}] Storage size mismatch: fast={fast_storage_size}, "
                                 f"normal={normal_storage_size} (may cause nondeterministic memory usage)")

                # Verify storage offset matches
                self.assertEqual(fast_result.storage_offset(), normal_result.storage_offset(),
                                 f"[{desc}] Storage offset mismatch in deterministic mode")

        finally:
            # Restore original state
            torch.use_deterministic_algorithms(original_deterministic)

    def test_empty_strided_npu_deterministic_repeatability(self, device="npu"):
        """
        Test that _empty_strided_npu produces repeatable results in deterministic mode.

        Multiple allocations with the same parameters should yield tensors with
        identical metadata and storage characteristics.
        """
        original_deterministic = torch.are_deterministic_algorithms_enabled()

        try:
            torch.use_deterministic_algorithms(True)

            sizes = (3, 4)
            strides = (8, 1)
            dtype = torch.float32

            # Allocate multiple times
            results = [
                torch_npu._C._empty_strided_npu(sizes, strides, dtype)
                for _ in range(5)
            ]

            # All results should have identical metadata
            first = results[0]
            for i, result in enumerate(results[1:], 1):
                self.assertEqual(result.shape, first.shape,
                                 f"Result {i} shape differs from first allocation")
                self.assertEqual(result.stride(), first.stride(),
                                 f"Result {i} stride differs from first allocation")
                self.assertEqual(result.storage().size(), first.storage().size(),
                                 f"Result {i} storage size differs from first allocation")

        finally:
            torch.use_deterministic_algorithms(original_deterministic)


if __name__ == "__main__":
    run_tests()
