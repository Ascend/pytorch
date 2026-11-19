import sympy
import torch
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from unittest import mock

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.kernel import mm as mm_kernel
from torch_npu._inductor.kernel.mm_common import NUM_CUBE_CORES

from testutils import TestUtils


def _cdiv(a, b):
    return (a + b - 1) // b


def _call_grid(grid_fn, *args, **kwargs):
    """Grid functions are wrapped by SymbolicGridFn; unwrap when needed."""
    return getattr(grid_fn, "fn", grid_fn)(*args, **kwargs)


class TestNpuMmKernelHelpers(TestUtils):
    """Assert the tiling generators directly; they are pure functions of the shape."""

    def test_block_candidates(self):
        # Padding eliminating M divisors come first, coarse N/K tiles fill the rest.
        self.assertIn(100, mm_kernel._gen_block_m_candidates(200, 256))
        self.assertEqual(mm_kernel._gen_block_m_candidates(256, 256), [256, 128, 64, 32])
        self.assertEqual(mm_kernel._gen_block_m_candidates(8, 256), [16])
        for n, expected in [(100, [112, 64, 32]), (16, [16]), (17, [32])]:
            self.assertEqual(mm_kernel._gen_block_n_candidates(n, 256), expected)
        self.assertEqual(mm_kernel._gen_block_n_candidates(8, 8), [8])
        self.assertEqual(mm_kernel._gen_block_k_candidates(204), [256, 128, 64])
        self.assertEqual(mm_kernel._gen_block_k_candidates(7), [16])
        for k in (64, 128, 204, 1000):
            cands = mm_kernel._gen_block_k_candidates(k)
            self.assertEqual(cands, sorted(cands, reverse=True))
            self.assertTrue(all(16 <= c <= 256 for c in cands))

    @parametrize("shape", [(128, 128, 128), (200, 100, 204), (16, 16, 16)])
    def test_mm_configs(self, shape):
        m, n, k = shape
        configs = mm_kernel._get_npu_mm_configs(m, n, k)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertEqual(cfg["num_stages"], 4)
            self.assertEqual(cfg["num_warps"], 4)
            self.assertEqual(cfg["ACC_TYPE"], "tl.float32")
            self.assertEqual(cfg["MASKED_M"], m % cfg["BLOCK_M"] != 0)
            self.assertEqual(cfg["MASKED_N"], n % cfg["BLOCK_N"] != 0)
            self.assertEqual(cfg["MASKED_K"], k % cfg["BLOCK_K"] != 0)
            # L0C cannot hold more than 64K fp32 accumulators.
            self.assertLessEqual(cfg["BLOCK_M"] * cfg["BLOCK_N"], 65536)

    def test_persistent_and_resident_configs(self):
        configs = mm_kernel._get_npu_persistent_mm_configs(512, 512, 512)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertEqual(cfg["WIDTH"], cfg["GROUP_M"] * cfg["NUM_BLOCKS_N"])
            self.assertEqual(cfg["NUM_BLOCKS"], cfg["NUM_BLOCKS_M"] * cfg["NUM_BLOCKS_N"])
            self.assertTrue(1 <= cfg["NUM_SMS"] <= NUM_CUBE_CORES)
            self.assertEqual(cfg["NUM_TILES_PER_PROGRAM"], _cdiv(cfg["NUM_BLOCKS"], cfg["NUM_SMS"]))
            self.assertEqual(cfg["NUM_BLOCKS_M"] % cfg["GROUP_M"], 0)
        # Dynamic shapes have no tiling to offer.
        s = sympy.Symbol("s0", integer=True, positive=True)
        for args in [(s, 512, 512), (512, s, 512), (512, 512, s)]:
            self.assertEqual(mm_kernel._get_npu_persistent_mm_configs(*args), [])

        # B[K, N] fits L0B, so N and K are fully loaded and M is streamed.
        configs = mm_kernel._get_npu_resident_b_mm_configs(256, 64, 64, dtype=torch.float16)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertEqual((cfg["BLOCK_N"], cfg["BLOCK_K"]), (64, 64))
            self.assertFalse(cfg["MASKED_N"] or cfg["MASKED_K"])
            self.assertEqual(cfg["BLOCK_M"] % 16, 0)
            self.assertTrue(1 <= cfg["NUM_PROGS"] <= NUM_CUBE_CORES)
        # A[K, M] fits L0A, so M and K are fully loaded and N is streamed.
        configs = mm_kernel._get_npu_resident_a_mm_configs(64, 512, 64, dtype=torch.float16)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertEqual((cfg["BLOCK_M"], cfg["BLOCK_K"]), (64, 64))
            self.assertFalse(cfg["MASKED_M"] or cfg["MASKED_K"])
            self.assertTrue(cfg["MASKED_N"])
        # Too small an N, an operand too large for its buffer and an unaddressable
        # extent are all declined.
        for shape in [(256, 8, 64), (256, 64, 1024), (256, 1024, 16)]:
            self.assertEqual(mm_kernel._get_npu_resident_b_mm_configs(*shape, dtype=torch.float16), [])
        for shape in [(8, 512, 64), (64, 512, 1024), (1024, 512, 64)]:
            self.assertEqual(mm_kernel._get_npu_resident_a_mm_configs(*shape, dtype=torch.float16), [])

    def test_grids(self):
        mm_meta = {"BLOCK_M": 32, "BLOCK_N": 64}
        for args, expected in [((128, 256, mm_meta), (16, 1, 1)), ((129, 257, mm_meta), (25, 1, 1))]:
            self.assertEqual(_call_grid(mm_kernel.npu_mm_grid, *args, cdiv=_cdiv), expected)
        persistent_meta = {"BLOCK_M": 64, "BLOCK_N": 64, "NUM_SMS": NUM_CUBE_CORES}
        # The grid is capped at NUM_SMS for many tiles and one program per tile otherwise.
        self.assertEqual(_call_grid(mm_kernel.npu_persistent_mm_grid, 1024, 1024, persistent_meta,
                                    cdiv=_cdiv, min=min), (NUM_CUBE_CORES, 1, 1))
        self.assertEqual(_call_grid(mm_kernel.npu_persistent_mm_grid, 64, 64, persistent_meta,
                                    cdiv=_cdiv, min=min), (1, 1, 1))
        for grid_fn in (mm_kernel.npu_resident_b_mm_grid, mm_kernel.npu_resident_a_mm_grid):
            self.assertEqual(_call_grid(grid_fn, 1024, 1024, {"NUM_PROGS": 7}, cdiv=_cdiv, min=min),
                             (7, 1, 1))


class TestNpuMmCompile(TestUtils):
    def setUp(self):
        super().setUp()
        # Reset the process-wide dynamo cache so the NPU kernels are not skipped
        # via a silent eager fallback when other inductor tests run first.
        torch._dynamo.reset()
        self._prev_cache_size_limit = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = 256

    def tearDown(self):
        torch._dynamo.config.cache_size_limit = self._prev_cache_size_limit
        super().tearDown()

    def op_mm(self, mat1, mat2):
        return torch.mm(mat1, mat2)

    def op_mm_transposed_a(self, mat1, mat2):
        return torch.mm(mat1.t(), mat2)

    def op_mm_transposed_b(self, mat1, mat2):
        return torch.mm(mat1, mat2.t())

    def op_mm_relu(self, mat1, mat2):
        return torch.relu(torch.mm(mat1, mat2))

    def op_addmm(self, bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2)

    def _inputs(self, shape, dtype):
        # randn is scaled down so fp16 dot products do not overflow.
        return torch.randn(shape, dtype=getattr(torch, dtype), device="npu") * 0.05

    def _run(self, op, args, dtype):
        expected = op(*args)
        compiled = torch.compile(op, backend="inductor")
        out, codes = run_and_get_code(compiled, *args)
        atol = 1e-3 if dtype == "float32" else 1e-2
        self.assertEqual(out, expected, atol=atol, rtol=atol)
        # A silent eager fallback keeps this green while running no NPU kernel.
        self.assertIn("def call(", "\n".join(codes))

    @parametrize("shape", [(64, 64, 64), (128, 256, 128)])
    @parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_mm_cases(self, shape, dtype):
        m, k, n = shape
        self._run(self.op_mm, (self._inputs((m, k), dtype), self._inputs((k, n), dtype)), dtype)

    @parametrize("bias_shape", [(256,), (128, 256)])
    def test_addmm_cases(self, bias_shape):
        # A 1-D bias is broadcast over the rows, a 2-D bias is used as is.
        self._run(self.op_addmm, (self._inputs(bias_shape, "float16"),
                                  self._inputs((128, 64), "float16"),
                                  self._inputs((64, 256), "float16")), "float16")

    def test_transposed_epilogue_and_dynamic(self):
        # Both operands can arrive as a col-major view, with a fused epilogue.
        for op, shape1, shape2 in [(self.op_mm_transposed_a, (128, 64), (128, 96)),
                                   (self.op_mm_transposed_b, (64, 128), (96, 128)),
                                   (self.op_mm_relu, (64, 128), (128, 96))]:
            self._run(op, (self._inputs(shape1, "float16"), self._inputs(shape2, "float16")), "float16")
        # M below one fractal (masked path) and M far above N (tall path).
        for m in (1, 512):
            self._run(self.op_mm, (self._inputs((m, 64), "float32"),
                                   self._inputs((64, 8), "float32")), "float32")
        compiled = torch.compile(self.op_mm, backend="inductor", dynamic=True)
        for m in (64, 128):
            mat1 = self._inputs((m, 96), "float32")
            mat2 = self._inputs((96, 64), "float32")
            self.assertEqual(compiled(mat1, mat2), torch.mm(mat1, mat2), atol=1e-3, rtol=1e-3)

    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON",
                  force_disable_caches=True)
    def test_max_autotune_adds_template_choices(self):
        # max_autotune makes tuned_mm()/tuned_addmm() offer the NPU template
        # families; the runtime benchmark picks the winner, so only the result is
        # asserted while the coverage comes from the offering itself.
        mat1 = self._inputs((128, 128), "float16")
        mat2 = self._inputs((128, 128), "float16")
        bias = self._inputs((128,), "float16")
        offered = []
        orig_offer = mm_kernel.add_npu_triton_mm_choices

        def counting_offer(*args, **kwargs):
            offered.append(1)
            return orig_offer(*args, **kwargs)

        with mock.patch.object(mm_kernel, "add_npu_triton_mm_choices", counting_offer):
            self._run(self.op_mm, (mat1, mat2), "float16")
        self.assertTrue(offered, "tuned_mm never offered NPU template choices")
        self._run(self.op_addmm, (bias, mat1, mat2), "float16")


instantiate_parametrized_tests(TestNpuMmKernelHelpers)
instantiate_parametrized_tests(TestNpuMmCompile)

if __name__ == "__main__":
    run_tests()
