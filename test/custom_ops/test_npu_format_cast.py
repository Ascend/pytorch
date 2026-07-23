import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

torch.npu.config.allow_internal_format = True

# ACL format constants
ACL_FORMAT_ND         = 2
ACL_FORMAT_FRACTAL_NZ = 29


class TestNpuFormatCastAclnn(TestCase):
    """
    Verify npu_format_cast aclnn implementation for ND -> FRACTAL_NZ.

    Test groups:
      1. ND -> NZ: format ID and data correctness
      2. Inplace API (npu_format_cast_)
      3. Tensor overload (npu_format_cast(src, dst_tensor))
      4. Same-format no-op
      5. Autograd / backward (requires_grad propagation)
      6. Error cases
    """

    # ------------------------------------------------------------------ #
    # Group 1: ND -> NZ format conversion
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nd_to_nz_format_id_fp16(self):
        t = torch.rand(8, 16).half().npu()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_FRACTAL_NZ)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nd_to_nz_format_id_bf16(self):
        t = torch.rand(32, 64).bfloat16().npu()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_FRACTAL_NZ)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nd_to_nz_format_id_float32(self):
        t = torch.rand(16, 32).float().npu()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_FRACTAL_NZ)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nd_to_nz_shape_preserved_fp16(self):
        t = torch.rand(64, 128).half().npu()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nd_to_nz_non_aligned_shape(self):
        t = torch.rand(15, 17).half().npu()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_1d_nd_to_nz_fallback_to_nd(self):
        t = torch.empty(12).float().npu()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_ND)
        self.assertEqual(out.shape, t.shape)

    # ------------------------------------------------------------------ #
    # Group 1b: NZ -> ND format conversion (reverse)
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nz_to_nd_format_id_fp16(self):
        """NZ -> ND: format changes back to ND."""
        t = torch.rand(8, 16).half().npu()
        nz = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        back = torch_npu.npu_format_cast(nz, ACL_FORMAT_ND)
        self.assertEqual(torch_npu.get_npu_format(back), ACL_FORMAT_ND)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nz_to_nd_format_id_bf16(self):
        """NZ -> ND bf16."""
        t = torch.rand(32, 64).bfloat16().npu()
        nz = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        back = torch_npu.npu_format_cast(nz, ACL_FORMAT_ND)
        self.assertEqual(torch_npu.get_npu_format(back), ACL_FORMAT_ND)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nz_to_nd_format_id_float32(self):
        """NZ -> ND float32."""
        t = torch.rand(16, 32).float().npu()
        nz = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        back = torch_npu.npu_format_cast(nz, ACL_FORMAT_ND)
        self.assertEqual(torch_npu.get_npu_format(back), ACL_FORMAT_ND)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_nz_to_nd_shape_preserved(self):
        """NZ -> ND: logical shape unchanged."""
        t = torch.rand(64, 128).half().npu()
        nz = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        back = torch_npu.npu_format_cast(nz, ACL_FORMAT_ND)
        self.assertEqual(back.shape, (64, 128))

    # ------------------------------------------------------------------ #
    # Group 2: Inplace API (npu_format_cast_)
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_inplace_nd_to_nz_format_id(self):
        t = torch.rand(16, 32).half().npu()
        torch_npu.npu_format_cast_(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(t), ACL_FORMAT_FRACTAL_NZ)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_inplace_nz_to_nd_format_id(self):
        """Inplace NZ -> ND: format changes back."""
        t = torch.rand(15, 17).half().npu()
        torch_npu.npu_format_cast_(t, ACL_FORMAT_FRACTAL_NZ)
        torch_npu.npu_format_cast_(t, ACL_FORMAT_ND)
        self.assertEqual(torch_npu.get_npu_format(t), ACL_FORMAT_ND)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_inplace_roundtrip_data(self):
        """Inplace ND -> NZ -> ND: data preserved."""
        t = torch.rand(15, 17).half().npu()
        expected = t.cpu().clone()
        torch_npu.npu_format_cast_(t, ACL_FORMAT_FRACTAL_NZ)
        torch_npu.npu_format_cast_(t, ACL_FORMAT_ND)
        self.assertEqual(t.cpu(), expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_inplace_same_format_noop(self):
        """Inplace to same format: storage unchanged."""
        t = torch.rand(8, 16).half().npu()
        ptr_before = t.storage().data_ptr()
        torch_npu.npu_format_cast_(t, ACL_FORMAT_ND)
        self.assertEqual(t.storage().data_ptr(), ptr_before)

    # ------------------------------------------------------------------ #
    # Group 3: Tensor overload npu_format_cast(src, dst_tensor)
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_tensor_overload_adopts_dst_format(self):
        """npu_format_cast(src, dst) converts src to dst's format."""
        src = torch.rand(8, 16).half().npu()
        dst_ref = torch_npu.npu_format_cast(torch.rand(8, 16).half().npu(), ACL_FORMAT_FRACTAL_NZ)
        result = torch_npu.npu_format_cast(src, dst_ref)
        self.assertEqual(torch_npu.get_npu_format(result), ACL_FORMAT_FRACTAL_NZ)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_tensor_overload_roundtrip_data(self):
        """npu_format_cast(src, dst) round-trip: data preserved."""
        src = torch.rand(15, 17).half().npu()
        expected = src.cpu().clone()
        nz_ref = torch_npu.npu_format_cast(torch.rand(15, 17).half().npu(), ACL_FORMAT_FRACTAL_NZ)
        nd_ref = torch.rand(15, 17).half().npu()
        nz = torch_npu.npu_format_cast(src, nz_ref)
        back = torch_npu.npu_format_cast(nz, nd_ref)
        self.assertEqual(back.cpu(), expected)

    # ------------------------------------------------------------------ #
    # Group 4: Same-format no-op
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_same_format_nd_noop(self):
        """ND -> ND: same storage returned."""
        t = torch.rand(4, 8).half().npu()
        ptr_before = t.storage().data_ptr()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_ND)
        self.assertEqual(out.storage().data_ptr(), ptr_before)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_same_format_nz_noop(self):
        """NZ -> NZ: same storage returned."""
        t = torch_npu.npu_format_cast(torch.rand(8, 16).half().npu(), ACL_FORMAT_FRACTAL_NZ)
        ptr_before = t.storage().data_ptr()
        out = torch_npu.npu_format_cast(t, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.storage().data_ptr(), ptr_before)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_same_format_get_format_consistent(self):
        """After same-format cast, get_npu_format unchanged."""
        t = torch.rand(8, 16).half().npu()
        fmt = torch_npu.get_npu_format(t)
        out = torch_npu.npu_format_cast(t, fmt)
        self.assertEqual(torch_npu.get_npu_format(out), fmt)

    # ------------------------------------------------------------------ #
    # Group 5: Autograd / backward
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_backward_requires_grad_preserved(self):
        """requires_grad propagates through format cast."""
        a = torch.rand(4, 8).half().npu().requires_grad_(True)
        b = torch_npu.npu_format_cast(a, ACL_FORMAT_FRACTAL_NZ)
        self.assertTrue(b.requires_grad)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_backward_same_format_noop(self):
        """Same-format cast (no-op path) also supports backward."""
        a = torch.rand(4, 8).float().npu().requires_grad_(True)
        ori_fmt = torch_npu.get_npu_format(a)
        b = torch_npu.npu_format_cast(a, ori_fmt)
        b.sum().backward()
        self.assertIsNotNone(a.grad)

    # ------------------------------------------------------------------ #
    # Group 6: Error cases
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_noncontiguous_with_internal_format_fallback(self):
        """910B/910_93: non-contig internal-format tensor falls back to aclop."""
        nz = torch_npu.npu_format_cast(torch.rand(16, 32).half().npu(), ACL_FORMAT_FRACTAL_NZ)
        nz_t = nz.transpose(0, 1)
        out = torch_npu.npu_format_cast(nz_t, ACL_FORMAT_ND)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_ND)

    @SupportedDevices(['Ascend950'])
    def test_noncontiguous_with_internal_format_raises(self):
        """950 (aclnn-only): non-contig internal-format tensor raises RuntimeError."""
        nz = torch_npu.npu_format_cast(torch.rand(16, 32).half().npu(), ACL_FORMAT_FRACTAL_NZ)
        nz_t = nz.transpose(0, 1)
        with self.assertRaises(RuntimeError):
            torch_npu.npu_format_cast(nz_t, ACL_FORMAT_ND)

    # ------------------------------------------------------------------ #
    # Group 7: View guard — 2D column-vector (Nx1) fallback to aclop
    # ------------------------------------------------------------------ #
    # When a 2D contiguous tensor is a view (storage shape != current shape)
    # and the target is FRACTAL_NZ, the aclnn path is skipped to avoid
    # downstream precision issues. The aclop fallback is exercised here.
    # Example: storage [1, N] viewed as [N, 1] (Nx1 column vector).

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_2d_view_nx1_nd_to_nz_format_id(self):
        """View [1, N] -> [N, 1]: format cast to NZ still succeeds."""
        # Create storage [1, N], then view as [N, 1]
        N = 16
        t = torch.rand(1, N).half().npu()
        t_view = t.t()  # contiguous, Nx1 column vector
        self.assertTrue(t_view.is_contiguous())

        out = torch_npu.npu_format_cast(t_view, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, (N, 1))
        self.assertEqual(out.storage()[1], t_view.storage()[1])


class TestNpuFormatCastDtypeParam(TestCase):
    """
    Tests for npu_format_cast with customize_dtype parameter.
    Default C0=16; customize_dtype=INT32 overrides to C0=8.
    """

    ACL_FORMAT_ND = 2
    ACL_FORMAT_FRACTAL_NZ = 29

    # dtype enum values (matching npu_native_functions.yaml / aclDataType)
    DTYPE_INT32 = 3

    def _expected_nz_storage_bytes(self, shape, c0, element_size):
        m = shape[-2]
        n = shape[-1]
        batch = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
        return batch * ((n + c0 - 1) // c0) * ((m + 15) // 16) * 16 * c0 * element_size

    # ------------------------------------------------------------------ #
    # Baseline: fp16/float32 default to C0=16; int8 defaults to C0=32.
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_fp16_nd_to_nz(self):
        """fp16 ND -> NZ, default C0=16."""
        t = torch.rand(16, 32).half().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_float32_nd_to_nz(self):
        """float32 ND -> NZ, default C0=16."""
        t = torch.rand(16, 32).float().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_int8_nd_to_nz(self):
        """int8 ND -> NZ, default C0=32."""
        t = torch.randint(-128, 127, (32, 64), dtype=torch.int8).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_int8_non_aligned(self):
        """int8 non-aligned shape, default C0=32."""
        t = torch.randint(-128, 127, (15, 17), dtype=torch.int8).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_fp16_default_c0_16_storage_size(self):
        """fp16 non-aligned shape uses default C0=16."""
        shape = (15, 17)
        t = torch.rand(*shape).half().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.untyped_storage().size(),
                         self._expected_nz_storage_bytes(shape, 16, 2))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_int8_default_c0_32_storage_size(self):
        """int8 non-aligned shape uses default C0=32 in infer shape."""
        shape = (15, 33)
        t = torch.randint(-128, 127, shape, dtype=torch.int8).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.untyped_storage().size(),
                         self._expected_nz_storage_bytes(shape, 32, 1))

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_npu_format_cast_int8_default_c0_32_storage_size_3d(self):
        """3D int8 infer shape preserves batch dims and uses C0=32."""
        shape = (3, 15, 33)
        t = torch.randint(-128, 127, shape, dtype=torch.int8).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.untyped_storage().size(),
                         self._expected_nz_storage_bytes(shape, 32, 1))

    # ------------------------------------------------------------------ #
    # int32 with customize_dtype=INT32 to override default C0=16 to C0=8:
    # Without customize_dtype, 4-bit types default to C0=16 (FORMAT_REAL_TO_FAKE).
    # Explicitly passing customize_dtype=INT32 bypasses that and uses C0=8.
    # (910B only: 950 does not support customize_dtype)
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_int32_customize_dtype_nd_to_nz(self):
        """int32 with explicit customize_dtype=INT32: C0=8 ND -> FRACTAL_NZ."""
        t = torch.randint(0, 100, (32, 32), dtype=torch.int32).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_int32_customize_dtype_non_aligned(self):
        """int32 with explicit customize_dtype=INT32: C0=8 non-aligned shape."""
        t = torch.randint(0, 100, (15, 17), dtype=torch.int32).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_int32_customize_dtype_3d(self):
        """int32 with explicit customize_dtype=INT32: C0=8 3D shape."""
        t = torch.randint(0, 100, (4, 16, 32), dtype=torch.int32).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    # ------------------------------------------------------------------ #
    # Storage size verification: explicitly verify C0 via storage bytes.
    # FRACTAL_NZ storage shape = (ceil(N/C0), ceil(M/16), 16, C0)
    # Different C0 values produce different storage sizes for non-aligned shapes.
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_c0_8_storage_size_2d(self):
        """Verify C0=8 storage size for (15, 17) int32.

        FRACTAL_NZ with C0=8: (ceil(17/8), ceil(15/16), 16, 8) = (3,1,16,8)
        = 384 elements * 4 bytes = 1536 bytes.
        """
        t = torch.randint(0, 100, (15, 17), dtype=torch.int32).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        # C0=8: ceil(17/8)*ceil(15/16)*16*8 * 4bytes = 3*1*128 * 4 = 1536
        expected_bytes = self._expected_nz_storage_bytes((15, 17), 8, 4)
        self.assertEqual(out.untyped_storage().size(), expected_bytes)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_c0_8_vs_default_c0_16_storage_differs(self):
        """C0=8 (customize_dtype=INT32) vs default C0=16 produce different storage.

        For (15, 17) int32:
          Default C0=16: (ceil(17/16), ceil(15/16), 16, 16) = 512 elements * 4 = 2048 bytes
          customize_dtype=INT32 C0=8: (ceil(17/8), ceil(15/16), 16, 8) = 384 elements * 4 = 1536 bytes
        """
        t = torch.randint(0, 100, (15, 17), dtype=torch.int32).npu()
        out_default = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        out_c0_8 = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        # Default C0=16 should be larger than C0=8
        self.assertGreater(out_default.untyped_storage().size(),
                           out_c0_8.untyped_storage().size())

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_default_c0_16_storage_size(self):
        """Verify default C0=16 storage size for (15, 17) int32 without customize_dtype.

        Default C0=16: (ceil(17/16), ceil(15/16), 16, 16) = (2,1,16,16)
        = 512 elements * 4 bytes = 2048 bytes.
        """
        t = torch.randint(0, 100, (15, 17), dtype=torch.int32).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        # Default C0=16: ceil(17/16)*ceil(15/16)*16*16 * 4bytes = 2*1*256*4 = 2048
        expected_bytes = self._expected_nz_storage_bytes((15, 17), 16, 4)
        self.assertEqual(out.untyped_storage().size(), expected_bytes)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast_c0_8_storage_size_3d(self):
        """Verify C0=8 storage size for 3D (4, 15, 17) int32.

        3D FRACTAL_NZ: batch dims preserved, last 2 dims follow NZ layout.
        C0=8: 4 * ceil(17/8) * ceil(15/16) * 16 * 8 = 4*3*1*16*8 = 1536 elements * 4 = 6144 bytes.
        """
        t = torch.randint(0, 100, (4, 15, 17), dtype=torch.int32).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        # 4 * ceil(17/8) * ceil(15/16) * 16 * 8 * 4bytes = 4*3*1*128*4 = 6144
        expected_bytes = self._expected_nz_storage_bytes((4, 15, 17), 8, 4)
        self.assertEqual(out.untyped_storage().size(), expected_bytes)


class TestFormatCast(TestCase):
    ACL_FORMAT_ND = 2
    ACL_FORMAT_FRACTAL_NZ = 29

    # ------------------------------------------------------------------ #
    # 2D: ND -> FRACTAL_NZ
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_2d_nd_to_nz_format_id(self):
        """2D fp16 ND -> FRACTAL_NZ format ID."""
        t = torch.rand(16, 32).half().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_2d_nd_to_nz_bf16(self):
        """2D bf16 ND -> FRACTAL_NZ."""
        t = torch.rand(16, 32).bfloat16().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_2d_nd_to_nz_int8(self):
        """2D int8 ND -> FRACTAL_NZ."""
        t = torch.randint(-128, 127, (32, 64), dtype=torch.int8).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_2d_nd_to_nz_large_shape(self):
        """2D large shape ND -> FRACTAL_NZ."""
        t = torch.rand(256, 512).half().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    # ------------------------------------------------------------------ #
    # 3D: ND -> FRACTAL_NZ
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_3d_nd_to_nz_format_id(self):
        """3D fp16 ND -> FRACTAL_NZ format ID."""
        t = torch.rand(4, 16, 32).half().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_3d_nd_to_nz_non_aligned(self):
        """3D non-aligned shape ND -> FRACTAL_NZ."""
        t = torch.rand(3, 15, 17).half().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_3d_nd_to_nz_bf16(self):
        """3D bf16 ND -> FRACTAL_NZ."""
        t = torch.rand(2, 16, 32).bfloat16().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_3d_nd_to_nz_float32(self):
        """3D float32 ND -> FRACTAL_NZ."""
        t = torch.rand(2, 16, 32).float().npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_3d_nd_to_nz_int8(self):
        """3D int8 ND -> FRACTAL_NZ."""
        t = torch.randint(-128, 127, (4, 16, 32), dtype=torch.int8).npu()
        out = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(torch_npu.get_npu_format(out), self.ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(out.shape, t.shape)


class TestNpuFormatCastPrecision(TestCase):
    """
    Precision tests for npu_format_cast via round-trip verification.

    ND -> FRACTAL_NZ -> ND: compare round-trip result against original data.
    (Direct storage comparison is not possible because untyped_storage().copy_()
    triggers internal TransData which fails on current CANN versions.)
    """

    def setUp(self):
        try:
            torch.npu.synchronize()
        except RuntimeError:
            pass

    ACL_FORMAT_ND = 2
    ACL_FORMAT_FRACTAL_NZ = 29

    DTYPE_INT32 = 3

    def _verify_roundtrip_nz(self, t_npu):
        """ND -> NZ -> ND round-trip, compare with original.

        Note: save expected data BEFORE format_cast because aclnnNpuFormatCast
        may corrupt the source tensor's buffer (CANN receives a non-const pointer
        via const_cast in ConvertType and writes to it during conversion).
        """
        expected = t_npu.cpu().clone()
        nz = torch_npu.npu_format_cast(t_npu, self.ACL_FORMAT_FRACTAL_NZ)
        back = torch_npu.npu_format_cast(nz, self.ACL_FORMAT_ND)
        self.assertEqual(torch_npu.get_npu_format(back), self.ACL_FORMAT_ND)
        self.assertEqual(back.shape, t_npu.shape)
        self.assertEqual(back.cpu(), expected)

    # ------------------------------------------------------------------ #
    # fp16 round-trip
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_fp16_2d_aligned(self):
        t = torch.rand(16, 32).half().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_fp16_2d_non_aligned(self):
        t = torch.rand(15, 17).half().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_fp16_3d(self):
        t = torch.rand(4, 16, 32).half().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_fp16_3d_non_aligned(self):
        t = torch.rand(3, 15, 17).half().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_fp16_large(self):
        t = torch.rand(256, 512).half().npu()
        self._verify_roundtrip_nz(t)

    # ------------------------------------------------------------------ #
    # bf16 round-trip
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_bf16_2d(self):
        t = torch.rand(16, 32).bfloat16().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_bf16_2d_non_aligned(self):
        t = torch.rand(15, 17).bfloat16().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_bf16_3d(self):
        t = torch.rand(3, 15, 17).bfloat16().npu()
        self._verify_roundtrip_nz(t)

    # ------------------------------------------------------------------ #
    # float32 round-trip
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_float32_2d(self):
        t = torch.rand(16, 32).float().npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_float32_3d(self):
        t = torch.rand(2, 16, 32).float().npu()
        self._verify_roundtrip_nz(t)

    # ------------------------------------------------------------------ #
    # int8 round-trip
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_int8_2d(self):
        t = torch.randint(-128, 127, (32, 64), dtype=torch.int8).npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_int8_2d_non_aligned(self):
        t = torch.randint(-128, 127, (15, 17), dtype=torch.int8).npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_int8_3d(self):
        t = torch.randint(-128, 127, (4, 16, 32), dtype=torch.int8).npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_int8_3d_non_aligned(self):
        t = torch.randint(-128, 127, (3, 15, 17), dtype=torch.int8).npu()
        self._verify_roundtrip_nz(t)

    @SupportedDevices(['Ascend910B', 'Ascend910_93', 'Ascend950'])
    def test_roundtrip_nz_reverse_direction_uses_format_sensitive_cache(self):
        t = torch.arange(16 * 32, dtype=torch.float16).reshape(16, 32).npu()
        self._verify_roundtrip_nz(t)

    # ------------------------------------------------------------------ #
    # int32 default C0=16 round-trip
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_roundtrip_nz_int32_default_c0_16(self):
        t = torch.randint(0, 100, (15, 17), dtype=torch.int32).npu()
        expected = t.cpu().clone()
        nz = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ)
        back = torch_npu.npu_format_cast(nz, self.ACL_FORMAT_ND)
        self.assertEqual(back.cpu(), expected)

    # ------------------------------------------------------------------ #
    # int32 with customize_dtype=INT32 (C0=8) round-trip on 910B
    # ------------------------------------------------------------------ #

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_roundtrip_nz_int32_c0_8_aligned(self):
        t = torch.randint(0, 100, (16, 8), dtype=torch.int32).npu()
        expected = t.cpu().clone()
        nz = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        back = torch_npu.npu_format_cast(nz, self.ACL_FORMAT_ND)
        self.assertEqual(back.cpu(), expected)

    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_roundtrip_nz_int32_c0_8_non_aligned(self):
        t = torch.randint(0, 100, (15, 17), dtype=torch.int32).npu()
        expected = t.cpu().clone()
        nz = torch_npu.npu_format_cast(t, self.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=self.DTYPE_INT32)
        back = torch_npu.npu_format_cast(nz, self.ACL_FORMAT_ND)
        self.assertEqual(back.cpu(), expected)


class TestZFormatCastOriginal(TestCase):
    ACL_FORMAT_NC1HWC0 = 3

    def supported_op_exec(self, input1):
        m = torch.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        output = m(input1)
        return output.cpu().detach()

    def custom_op_exec(self, input1, acl_format):
        output = torch_npu.npu_format_cast(input1, acl_format)
        return output.cpu().detach()

    # Ascend950 does not support this 4D NC1HWC0 format_cast case.
    @SupportedDevices(['Ascend910B', 'Ascend910_93'])
    def test_npu_format_cast(self, device="npu"):
        """Original test: NCHW -> NC1HWC0 data correctness."""
        item = [np.float16, 0, (2, 2, 4, 4)]
        _, npu_input = create_common_tensor(item, -1, 1)

        supported_output = self.supported_op_exec(npu_input)
        custom_output = self.custom_op_exec(npu_input, self.ACL_FORMAT_NC1HWC0)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
