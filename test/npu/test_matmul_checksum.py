import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices
from torch_npu.testing.testcase import TestCase, run_tests

import torch_npu.asd.checksum as matmul_checksum

# Shapes validated against the operator: K >= 64 (small-K bf16 inputs hit a
# known operator-side threshold coverage issue) and multi-segment N.
ABFT_CLEAN_SHAPES = [
    (256, 1024, 512),   # M % 8 == 0, two column segments
    (33, 100, 67),      # M % 8 == 1
    (12, 64, 512),      # M % 8 == 4, partial row group
    (7, 64, 9),         # M % 8 == 7, single byte
]


class TestMatmulChecksumInputs(TestCase):
    # Input validation only, no operator dependency: runs on every NPU env.

    def test_cpu_tensor_raises_type_error(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 8)
        c = torch.matmul(a, b)
        with self.assertRaises(TypeError):
            torch_npu.matmul_checksum(a, b, c)


class TestMatmulChecksumPyFallback(TestCase):
    # Pure-PyTorch fallback and the dispatch contract: covered by standard CI
    # environments, which do not ship the aclnnMatmulAbftVerify symbols. The
    # fallback is forced by flipping the availability cache (same internal
    # patching pattern as test/utils/test_asd_detector.py).

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        self._cache_backup = matmul_checksum._abft_available_cache
        matmul_checksum._abft_available_cache = False

    def tearDown(self):
        matmul_checksum._abft_available_cache = self._cache_backup
        super().tearDown()

    def _make(self, m, k, n, dtype):
        a = torch.randn(m, k, dtype=dtype).npu()
        b = torch.randn(k, n, dtype=dtype).npu()
        c = torch.matmul(a, b)
        return a, b, c

    def test_clean_result_no_anomaly(self):
        for dtype in (torch.bfloat16, torch.float32):
            a, b, c = self._make(64, 128, 32, dtype)
            ret = torch_npu.matmul_checksum(a, b, c)
            self.assertEqual(ret.dim(), 0)
            self.assertEqual(ret.dtype, torch.bool)
            self.assertFalse(ret.item())

    def test_corrupted_result_detected(self):
        a, b, c = self._make(64, 128, 32, torch.bfloat16)
        c[5, :] += 100.0
        self.assertTrue(torch_npu.matmul_checksum(a, b, c).item())

    def test_batched_3d_falls_back(self):
        a = torch.randn(4, 16, 32, dtype=torch.bfloat16).npu()
        b = torch.randn(4, 32, 8, dtype=torch.bfloat16).npu()
        c = torch.matmul(a, b)
        ret = torch_npu.matmul_checksum(a, b, c)
        self.assertFalse(ret.item())

    def test_dtype_mismatch_falls_back(self):
        a = torch.randn(8, 16, dtype=torch.float32).npu()
        b = torch.randn(16, 8, dtype=torch.bfloat16).npu()
        c = torch.matmul(a, b.to(torch.float32)).to(torch.bfloat16)
        self.assertFalse(torch_npu.matmul_checksum(a, b, c).item())

    def test_c_lower_precision_falls_back(self):
        a = torch.randn(8, 64, dtype=torch.float32).npu()
        b = torch.randn(64, 8, dtype=torch.float32).npu()
        c = torch.matmul(a, b).to(torch.bfloat16)
        self.assertFalse(torch_npu.matmul_checksum(a, b, c).item())

    def test_abft_path_applicable(self):
        a, b, c = self._make(64, 128, 32, torch.bfloat16)
        self.assertTrue(matmul_checksum._abft_path_applicable(a, b, c))

        ab = torch.randn(4, 16, 32, dtype=torch.bfloat16).npu()
        bb = torch.randn(4, 32, 8, dtype=torch.bfloat16).npu()
        self.assertFalse(matmul_checksum._abft_path_applicable(ab, bb, torch.matmul(ab, bb)))

        am = torch.randn(8, 16, dtype=torch.float32).npu()
        bm = torch.randn(16, 8, dtype=torch.bfloat16).npu()
        cm = torch.matmul(am, bm.to(torch.float32)).to(torch.bfloat16)
        self.assertFalse(matmul_checksum._abft_path_applicable(am, bm, cm))

        ac = torch.randn(8, 64, dtype=torch.float32).npu()
        bc = torch.randn(64, 8, dtype=torch.float32).npu()
        cc = torch.matmul(ac, bc).to(torch.bfloat16)
        self.assertFalse(matmul_checksum._abft_path_applicable(ac, bc, cc))

        # c may upgrade to fp32 over bf16 inputs (op contract allows it).
        ah = torch.randn(8, 64, dtype=torch.bfloat16).npu()
        bh = torch.randn(64, 8, dtype=torch.bfloat16).npu()
        ch = torch.matmul(ah, bh).to(torch.float32)
        self.assertTrue(matmul_checksum._abft_path_applicable(ah, bh, ch))

        as_ = torch.randn(8, 64, dtype=torch.bfloat16).npu()
        bs_ = torch.randn(32, 8, dtype=torch.bfloat16).npu()
        cs_ = torch.randn(8, 8, dtype=torch.bfloat16).npu()
        self.assertFalse(matmul_checksum._abft_path_applicable(as_, bs_, cs_))


class TestMatmulChecksumAbft(TestCase):
    # V-ABFT op path. The operator ships via the CANN ops-ras vendors package
    # instead of the base toolkit, so gate on a live availability probe
    # (not the CANN version) and skip when the environment lacks it.

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        if not matmul_checksum._npu_matmul_abft_verify_available():
            self.skipTest("npu_matmul_abft_verify op is unavailable")

    def _make(self, m, k, n, dtype):
        a = torch.randn(m, k, dtype=dtype).npu()
        b = torch.randn(k, n, dtype=dtype).npu()
        c = torch.matmul(a, b)
        return a, b, c

    @SupportedDevices(['Ascend910B', 'Ascend910C'])
    def test_clean_result_no_anomaly(self):
        for dtype in (torch.bfloat16, torch.float32):
            for (m, k, n) in ABFT_CLEAN_SHAPES:
                a, b, c = self._make(m, k, n, dtype)
                ret = torch_npu.matmul_checksum(a, b, c)
                self.assertEqual(ret.dim(), 0)
                self.assertEqual(ret.dtype, torch.bool)
                self.assertFalse(ret.item())
        # FP32 MOE-gate style shape.
        a, b, c = self._make(256, 4096, 8, torch.float32)
        self.assertFalse(torch_npu.matmul_checksum(a, b, c).item())

    @SupportedDevices(['Ascend910B', 'Ascend910C'])
    def test_corrupted_result_detected(self):
        a, b, c = self._make(256, 1024, 512, torch.bfloat16)
        c2 = c.clone()
        c2[3, 300] += 100.0  # row 3, second column segment
        self.assertTrue(torch_npu.matmul_checksum(a, b, c2).item())

        c2 = c.clone()
        c2[11, :] += 10.0  # row 11 lands in a partial row group when M=12
        self.assertTrue(torch_npu.matmul_checksum(a[:12], b, c2[:12]).item())

        a32, b32, c32 = self._make(64, 128, 64, torch.float32)
        c32[0, :] += 10.0
        self.assertTrue(torch_npu.matmul_checksum(a32, b32, c32).item())

    @SupportedDevices(['Ascend910B', 'Ascend910C'])
    def test_bf16_ab_fp32_c(self):
        # Mixed dtypes per the op contract: bf16 inputs with fp32 output.
        a = torch.randn(256, 1024, dtype=torch.bfloat16).npu()
        b = torch.randn(1024, 512, dtype=torch.bfloat16).npu()
        c = torch.matmul(a, b).to(torch.float32)
        self.assertFalse(torch_npu.matmul_checksum(a, b, c).item())

        c2 = c.clone()
        c2[3, 300] += 100.0  # row 3, second column segment
        self.assertTrue(torch_npu.matmul_checksum(a, b, c2).item())

    @SupportedDevices(['Ascend910B', 'Ascend910C'])
    def test_non_contiguous_b_op_path(self):
        a = torch.randn(8, 16, dtype=torch.bfloat16).npu()
        b = torch.randn(64, 16, dtype=torch.bfloat16).npu().t()  # [16, 64] non-contiguous
        c = torch.matmul(a, b)
        self.assertFalse(torch_npu.matmul_checksum(a, b, c).item())
        c2 = c.clone()
        c2[2, 5] += 50.0
        self.assertTrue(torch_npu.matmul_checksum(a, b, c2).item())


if __name__ == '__main__':
    run_tests()
