import os
import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

os.environ["PYTORCH_NPU_ALLOC_CONF"] = 'expandable_segments:True'

class  TestTensorToPreserveFormat(TestCase):
    def _make_non_contiguous_tensors(self):
        """
        Generate various types of non-contiguous tensors, returns list:
          (name, cpu_tensor, is_non_overlapping_and_dense)
        """
        tensors = []
        # 1. transpose 2D
        t = torch.randn(4, 6).t()
        tensors.append(("transpose_2d", t, True))
        # 2. permute 4D
        t = torch.randn(2, 3, 4, 5).permute(3, 1, 0, 2)
        tensors.append(("permute_4d", t, True))
        # 3. slice dim0 (step > 1)
        t = torch.randn(8, 5)[::2]
        tensors.append(("slice_dim0", t, True))
        # 4. slice dim1 (step > 1)
        t = torch.randn(4, 8)[:, ::2]
        tensors.append(("slice_dim1", t, True))
        # 5. narrow
        t = torch.randn(6, 8).narrow(1, 1, 5)
        tensors.append(("narrow", t, True))
        # 6. select (reduces dim)
        t = torch.randn(3, 5, 4).select(1, 2)
        tensors.append(("select_dim1", t, True))
        # 7. 1D non-contiguous (as_strided stride > 1)
        t = torch.randn(10).as_strided((4,), (2,))
        tensors.append(("1d_as_strided", t, True))
        # 8. expand 2D (stride=0)
        t = torch.randn(1, 5).expand(4, 5)
        tensors.append(("expand_2d", t, False))
        # 9. expand 3D (stride=0)
        t = torch.randn(2, 1, 4).expand(2, 3, 4)
        tensors.append(("expand_3d", t, False))
        # 10. as_strided with overlap (elements overlap, non-overlapping=False)
        t = torch.randn(12).as_strided((3, 3), (4, 1))
        tensors.append(("as_strided_overlap", t, False))

        return tensors

    def _verify_preserve_format_result(self, src, result, scenario_name):
        """Verify basic correctness of preserve_format result"""
        self.assertEqual(list(src.shape), list(result.shape))

    def _verify_stride_preserved(self, src, result, scenario_name, is_nod):
        """
        Verify stride behavior:
          - is_non_overlapping_and_dense=True -> stride should be preserved
          - is_non_overlapping_and_dense=False -> stride may differ (fallback to contiguous)
        """
        if is_nod:
            self.assertEqual(src.stride(), result.stride())
        else:
            pass

    # ================================================================
    # H2D scenario: CPU -> NPU
    # ================================================================

    def test_h2d_transpose_2d(self):
        """H2D: transpose 2D non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(4, 6).t()
        self.assertFalse(cpu_t.is_contiguous())


        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-transpose_2d")
        self.assertEqual(str(npu_t.device), "npu:0")
        self.assertEqual(cpu_t.stride(), npu_t.stride(),
                         "H2D-transpose_2d: stride not preserved")

    def test_h2d_permute_4d(self):
        """H2D: permute 4D non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(2, 3, 4, 5).permute(3, 1, 0, 2)
        self.assertFalse(cpu_t.is_contiguous())
        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-permute_4d")
        self.assertEqual(cpu_t.stride(), npu_t.stride(),
                         "H2D-permute_4d: stride not preserved")

    def _get_dense_strides(self, src):
        """Get dense storage strides"""
        dummy = torch.empty_like(src)
        return dummy.stride()

    def test_h2d_slice_dim0(self):
        """H2D: slice dim0 non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(8, 5)[::2]
        self.assertFalse(cpu_t.is_contiguous())


        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        expec_strides = self._get_dense_strides(cpu_t)
        self.assertEqual(expec_strides, npu_t.stride(),
                         "H2D-slice_dim0: stride not preserved")

    def test_h2d_slice_dim1(self):
        """H2D: slice dim1 non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(4, 8)[:, ::2]
        self.assertFalse(cpu_t.is_contiguous())


        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-slice_dim1")
        expec_strides = self._get_dense_strides(cpu_t)
        self.assertEqual(expec_strides, npu_t.stride(),
                         "H2D-slice_dim1: stride not preserved")

    def test_h2d_narrow(self):
        """H2D: narrow non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(6, 8).narrow(1, 1, 5)
        self.assertFalse(cpu_t.is_contiguous())


        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-narrow")
        expec_strides = self._get_dense_strides(cpu_t)
        self.assertEqual(expec_strides, npu_t.stride(),
                         "H2D-narrow: stride not preserved")

    def test_h2d_select(self):
        """H2D: select (reduces dim) non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(3, 5, 4).select(1, 2)
        self.assertFalse(cpu_t.is_contiguous())

        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-select")
        expec_strides = self._get_dense_strides(cpu_t)
        self.assertEqual(expec_strides, npu_t.stride(),
                         "H2D-select: stride not preserved")

    def test_h2d_1d_as_strided(self):
        """H2D: 1D as_strided non-contiguous -> preserve_format preserves stride"""
        cpu_t = torch.randn(10).as_strided((4,), (2,))
        self.assertFalse(cpu_t.is_contiguous())

        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-1d_as_strided")
        expec_strides = self._get_dense_strides(cpu_t)
        self.assertEqual(expec_strides, npu_t.stride(),
                         "H2D-1d_as_strided: stride not preserved")

    def test_h2d_expand_2d(self):
        """H2D: expand 2D (stride=0) -> preserve_format falls back to suggest_memory_format"""
        cpu_t = torch.randn(1, 5).expand(4, 5)
        self.assertFalse(cpu_t.is_contiguous())

        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-expand_2d")
        # expand stride is not preserved, falls back to contiguous
        self.assertTrue(npu_t.is_contiguous(),
                        "H2D-expand_2d: non-overlapping-and-dense should fall back to contiguous")

    def test_h2d_expand_3d(self):
        """H2D: expand 3D (stride=0) -> preserve_format falls back to suggest_memory_format"""
        cpu_t = torch.randn(2, 1, 4).expand(2, 3, 4)
        self.assertFalse(cpu_t.is_contiguous())


        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-expand_3d")
        self.assertTrue(npu_t.is_contiguous(),
                        "H2D-expand_3d: non-overlapping-and-dense should fall back to contiguous")

    def test_h2d_as_strided_overlap(self):
        """H2D: as_strided with overlap -> preserve_format falls back to suggest_memory_format"""
        cpu_t = torch.randn(12).as_strided((3, 3), (4, 1))


        npu_t = cpu_t.to("npu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(cpu_t, npu_t, "H2D-as_strided_overlap")
        # overlapping tensor falls back, result should be contiguous
        self.assertTrue(npu_t.is_contiguous(),
                        "H2D-as_strided_overlap: non-overlapping-and-dense should fall back to contiguous")

    # ================================================================
    # D2H scenario: NPU -> CPU
    # ================================================================

    def test_d2h_transpose_2d(self):
        """D2H: transpose 2D non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(4, 6, device="npu").t()
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-transpose_2d")
        self.assertEqual(npu_t.stride(), cpu_t.stride(),
                         "D2H-transpose_2d: stride not preserved")

    def test_d2h_permute_4d(self):
        """D2H: permute 4D non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(2, 3, 4, 5, device="npu").permute(3, 1, 0, 2)
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-permute_4d")
        self.assertEqual(npu_t.stride(), cpu_t.stride(),
                         "D2H-permute_4d: stride not preserved")

    def test_d2h_slice_dim0(self):
        """D2H: slice dim0 non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(8, 5, device="npu")[::2]
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-slice_dim0")
        expec_strides = self._get_dense_strides(npu_t)
        self.assertEqual(expec_strides, cpu_t.stride(),
                         "D2H-slice_dim0: stride not preserved")

    def test_d2h_slice_dim1(self):
        """D2H: slice dim1 non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(4, 8, device="npu")[:, ::2]
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-slice_dim1")
        expec_strides = self._get_dense_strides(npu_t)
        self.assertEqual(expec_strides, cpu_t.stride(),
                         "D2H-slice_dim1: stride not preserved")

    def test_d2h_narrow(self):
        """D2H: narrow non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(6, 8, device="npu").narrow(1, 1, 5)
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-narrow")
        expec_strides = self._get_dense_strides(npu_t)
        self.assertEqual(expec_strides, cpu_t.stride(),
                         "D2H-narrow: stride not preserved")

    def test_d2h_select(self):
        """D2H: select (reduces dim) non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(3, 5, 4, device="npu").select(1, 2)
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-select")
        expec_strides = self._get_dense_strides(npu_t)
        self.assertEqual(expec_strides, cpu_t.stride(),
                         "D2H-select: stride not preserved")

    def test_d2h_1d_as_strided(self):
        """D2H: 1D as_strided non-contiguous -> preserve_format preserves stride"""
        npu_t = torch.randn(10, device="npu").as_strided((4,), (2,))
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-1d_as_strided")
        expec_strides = self._get_dense_strides(npu_t)
        self.assertEqual(expec_strides, cpu_t.stride(),
                         "D2H-1d_as_strided: stride not preserved")

    def test_d2h_expand_2d(self):
        """D2H: expand 2D (stride=0) -> preserve_format falls back to suggest_memory_format"""
        npu_t = torch.randn(1, 5, device="npu").expand(4, 5)
        self.assertFalse(npu_t.is_contiguous())

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-expand_2d")
        self.assertTrue(cpu_t.is_contiguous(),
                        "D2H-expand_2d: non-overlapping-and-dense should fall back to contiguous")

    def test_d2h_expand_3d(self):
        """D2H: expand 3D (stride=0) -> preserve_format falls back to suggest_memory_format"""
        npu_t = torch.randn(2, 1, 4, device="npu").expand(2, 3, 4)
        self.assertFalse(npu_t.is_contiguous())
        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-expand_3d")
        self.assertTrue(cpu_t.is_contiguous(),
                        "D2H-expand_3d: non-overlapping-and-dense should fall back to contiguous")

    def test_d2h_as_strided_overlap(self):
        """D2H: as_strided with overlap -> preserve_format falls back to suggest_memory_format"""
        npu_t = torch.randn(12, device="npu").as_strided((3, 3), (4, 1))

        cpu_t = npu_t.to("cpu", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu_t, cpu_t, "D2H-as_strided_overlap")
        self.assertTrue(cpu_t.is_contiguous(),
                        "D2H-as_strided_overlap: non-overlapping-and-dense should fall back to contiguous")

    # ================================================================
    # D2D scenario: NPU:0 -> NPU:1 (different device_index)
    # ================================================================

    @skipIfUnsupportMultiNPU(2)
    def test_d2d_transpose_2d(self):
        """D2D: transpose 2D non-contiguous -> preserve_format preserves stride"""
        npu0_t = torch.randn(4, 6, device="npu:0").t()
        self.assertFalse(npu0_t.is_contiguous())

        npu1_t = npu0_t.to("npu:1", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu0_t, npu1_t, "D2D-transpose_2d")
        self.assertEqual(str(npu1_t.device), "npu:1")
        self.assertEqual(npu0_t.stride(), npu1_t.stride(),
                         "D2D-transpose_2d: stride not preserved")

    @skipIfUnsupportMultiNPU(2)
    def test_d2d_permute_4d(self):
        """D2D: permute 4D non-contiguous -> preserve_format preserves stride"""
        npu0_t = torch.randn(2, 3, 4, 5, device="npu:0").permute(3, 1, 0, 2)
        self.assertFalse(npu0_t.is_contiguous())

        npu1_t = npu0_t.to("npu:1", memory_format=torch.preserve_format)
        self._verify_preserve_format_result(npu0_t, npu1_t, "D2D-permute_4d")
        self.assertEqual(str(npu1_t.device), "npu:1")
        self.assertEqual(npu0_t.stride(), npu1_t.stride(),
                         "D2D-permute_4d: stride not preserved")

if __name__ == "__main__":
    run_tests()