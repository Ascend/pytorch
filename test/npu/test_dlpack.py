import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests
from version_mark import runIfVersion


@instantiate_tests
class TestDLPack(TestCase):

    @Dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)
    def test_dlpack_roundtrip_basic(self, dtype, device="npu"):
        """Test basic dlpack roundtrip: torch_npu tensor -> dlpack -> torch_npu tensor"""
        # Create original tensor
        if dtype == torch.bool:
            original = torch.randint(0, 2, (2, 3, 4), dtype=dtype, device=device)
        elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            original = torch.randint(-10, 10, (2, 3, 4), dtype=dtype, device=device)
        else:
            original = torch.randn(2, 3, 4, dtype=dtype, device=device)

        # Convert to dlpack
        dlpack_tensor = to_dlpack(original)

        # Convert back to torch_npu tensor
        restored = from_dlpack(dlpack_tensor)

        # Verify the roundtrip
        self.assertEqual(original.cpu(), restored.cpu())
        self.assertEqual(original.dtype, restored.dtype)
        self.assertEqual(original.device, restored.device)
        self.assertEqual(original.shape, restored.shape)
        self.assertEqual(original.stride(), restored.stride())

    @Dtypes(torch.half, torch.float, torch.double)
    def test_dlpack_roundtrip_different_shapes(self, dtype, device="npu"):
        """Test dlpack roundtrip with different tensor shapes"""
        shapes = [
            (1,),           # 1D tensor
            (5, 5),         # 2D tensor
            (2, 3, 4),      # 3D tensor
            (2, 2, 2, 2),   # 4D tensor
            (1, 1, 1, 1, 1)  # 5D tensor
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                original = torch.randn(shape, dtype=dtype, device=device)
                dlpack_tensor = to_dlpack(original)
                restored = from_dlpack(dlpack_tensor)

                self.assertEqual(original.cpu(), restored.cpu())
                self.assertEqual(original.shape, restored.shape)

    @Dtypes(torch.float)
    def test_dlpack_roundtrip_contiguous(self, dtype, device="npu"):
        """Test dlpack roundtrip with contiguous and non-contiguous tensors"""
        # Test contiguous tensor
        original_contiguous = torch.randn(4, 4, dtype=dtype, device=device)
        self.assertTrue(original_contiguous.is_contiguous())

        dlpack_tensor = to_dlpack(original_contiguous)
        restored = from_dlpack(dlpack_tensor)

        self.assertEqual(original_contiguous, restored)
        self.assertTrue(restored.is_contiguous())

        # Test non-contiguous tensor (transpose)
        original_non_contiguous = original_contiguous.t()
        self.assertFalse(original_non_contiguous.is_contiguous())

        dlpack_tensor = to_dlpack(original_non_contiguous)
        restored = from_dlpack(dlpack_tensor)

        self.assertEqual(original_non_contiguous, restored)
        self.assertEqual(original_non_contiguous.stride(), restored.stride())

    @Dtypes(torch.float)
    def test_dlpack_memory_sharing(self, dtype, device="npu"):
        """Test that dlpack shares memory with original tensor"""
        original = torch.randn(3, 3, dtype=dtype, device=device)
        original_data_ptr = original.data_ptr()

        # Convert to dlpack and back
        dlpack_tensor = to_dlpack(original)
        restored = from_dlpack(dlpack_tensor)

        # Check if memory is shared (data_ptr should be the same)
        self.assertEqual(original_data_ptr, restored.data_ptr())

        # Modify original tensor and check if restored tensor is also modified
        original.fill_(42.0)
        self.assertEqual(original, restored)

    @Dtypes(torch.bfloat16)
    def test_dlpack_bfloat16_support(self, dtype, device="npu"):
        """Test dlpack with bfloat16 data type"""
        original = torch.randn(3, 4, dtype=dtype, device=device)
        dlpack_tensor = to_dlpack(original)
        restored = from_dlpack(dlpack_tensor)

        self.assertEqual(original, restored)
        self.assertEqual(original.dtype, restored.dtype)

    @Dtypes(torch.float)
    def test_dlpack_cpu(self, dtype, device="cpu"):
        """Test that dlpack shares memory with original cpu tensor"""
        original = torch.randn(3, 3, dtype=dtype, device=device)
        original_data_ptr = original.data_ptr()

        # Convert to dlpack and back
        dlpack_tensor = to_dlpack(original)
        restored = from_dlpack(dlpack_tensor)

        # Check if memory is shared (data_ptr should be the same)
        self.assertEqual(original_data_ptr, restored.data_ptr())

    @runIfVersion(min="2.14")
    def test_from_dlpack_negative_strides(self):
        """NPU adaptation of upstream PR #188047.

        torch.from_dlpack() on a NumPy array with negative strides used to
        abort the process instead of raising a catchable Python exception.
        See https://github.com/pytorch/pytorch/issues/188023.
        """
        import numpy as np

        # 1-D negative stride
        a1 = np.arange(8.0)[::-1]
        with self.assertRaisesRegex(
            RuntimeError, "Storage size calculation overflowed"
        ):
            torch.from_dlpack(a1)

        # 2-D, one negative axis
        a2 = np.arange(12.0).reshape(3, 4)[:, ::-1]
        with self.assertRaisesRegex(
            RuntimeError, "Storage size calculation overflowed"
        ):
            torch.from_dlpack(a2)

    @runIfVersion(min="2.14")
    def test_from_dlpack_negative_strides_npu(self, device="npu"):
        """NPU-device variant of test_from_dlpack_negative_strides.

        NPU capsules are imported via at_npu::native::from_blob, whose
        computeStorageSize() noexcept was dropped to align with upstream
        pytorch#188062. A negative stride cannot be built on device (slice
        rejects negative steps), so the exported capsule's stride field is
        patched in place, and _npu_from_dlpack is called directly to skip
        the Python-level fallback to torch's implementation.
        """
        import ctypes

        from torch_npu._C import _npu_from_dlpack

        class DLTensor(ctypes.Structure):
            # DLPack DLTensor layout (first member of DLManagedTensor).
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int),
                ("ndim", ctypes.c_int),
                ("dtype_code", ctypes.c_uint8),
                ("dtype_bits", ctypes.c_uint8),
                ("dtype_lanes", ctypes.c_uint16),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64),
            ]

        capsule = to_dlpack(torch.arange(8.0, device=device))
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        dl_tensor = DLTensor.from_address(
            get_pointer(ctypes.py_object(capsule), b"dltensor")
        )
        dl_tensor.strides[0] = -1

        with self.assertRaisesRegex(
            RuntimeError, "Storage size calculation overflowed"
        ):
            _npu_from_dlpack(capsule)

    @runIfVersion(min="2.14")
    def test_from_blob_strides_overflow(self):
        """NPU adaptation of upstream BasicTest::TestForBlobStridesOverflow
        (aten/src/ATen/test/basic.cpp, PR #188062): an overflowing or
        mismatched stride must raise a catchable RuntimeError instead of
        terminating the process out of the noexcept computeStorageSize().
        """
        import numpy as np

        arr = np.zeros((8,), dtype=np.float32)
        ptr = arr.ctypes.data

        # Mismatched sizes/strides dimensionality throws on all builds.
        with self.assertRaisesRegex(
            RuntimeError, "must match dimensionality of strides"
        ):
            torch._from_blob(ptr, (2, 3), (1,), dtype=torch.float32)

        # A stride large enough to overflow the storage size computation
        # also throws; overflow checks are compiled out on mobile builds only.
        huge = torch.iinfo(torch.int64).max
        with self.assertRaisesRegex(
            RuntimeError, "Storage size calculation overflowed"
        ):
            torch._from_blob(ptr, (2,), (huge,), dtype=torch.float32)

    @runIfVersion(min="2.14")
    def test_from_blob_strides_overflow_npu(self, device="npu"):
        """NPU-device variant of test_from_blob_strides_overflow.

        The size checks run on the host before any device memory is wrapped,
        so from_blob with a valid NPU device pointer must raise the same
        catchable RuntimeError instead of crashing the process.
        """
        buf = torch.zeros(8, dtype=torch.float32, device=device)
        ptr = buf.data_ptr()

        # Mismatched sizes/strides dimensionality throws on all builds.
        with self.assertRaisesRegex(
            RuntimeError, "must match dimensionality of strides"
        ):
            torch._from_blob(ptr, (2, 3), (1,), dtype=torch.float32, device=device)

        # A stride large enough to overflow the storage size computation
        # also throws; overflow checks are compiled out on mobile builds only.
        huge = torch.iinfo(torch.int64).max
        with self.assertRaisesRegex(
            RuntimeError, "Storage size calculation overflowed"
        ):
            torch._from_blob(ptr, (2,), (huge,), dtype=torch.float32, device=device)


if __name__ == '__main__':
    run_tests()
