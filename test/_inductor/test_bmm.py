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
from torch_npu._inductor.kernel import bmm as bmm_kernel
from torch_npu._inductor.kernel.mm_common import (
    ACCUM_BYTES,
    DEFAULT_ELEM_BYTES,
    L0C_BUFFERING_KEY,
    L0C_BYTES,
    L0C_DOUBLE_BUFFERED,
    L0C_DOUBLE_BUFFERED_BYTES,
    L0C_SINGLE_BUFFERED,
    L1_BUFFER_COPIES,
    MMAD_K_FRACTAL,
    MMAD_M_FRACTAL,
    NUM_CUBE_CORES,
)

from testutils import TestUtils


def _cdiv(a, b):
    return (a + b - 1) // b


def _call_grid(grid_fn, *args, **kwargs):
    """Grid functions are wrapped by SymbolicGridFn; unwrap when needed."""
    return getattr(grid_fn, "fn", grid_fn)(*args, **kwargs)


class _FakeNode:
    """Minimal stand-in for an IR node: the dispatch gates read size only."""

    def __init__(self, size, stride=None):
        self._size = tuple(size)
        self._stride = tuple(stride if stride else size)

    def get_size(self):
        return self._size

    def get_stride(self):
        return self._stride


class _FakeLayout:
    def __init__(self, size, stride):
        self.size = tuple(size)
        self.stride = tuple(stride)


class TestNpuBmmKernelHelpers(TestUtils):
    """Assert the tiling generators directly; they need no device."""

    def test_capacity_split_and_wave_helpers(self):
        for dtype, expected in [(torch.float32, 4), (torch.float16, 2), (torch.bfloat16, 2),
                                (torch.int8, 1), (torch.float64, 8)]:
            self.assertEqual(bmm_kernel._elem_bytes(dtype), expected)
        self.assertEqual(bmm_kernel._elem_bytes(None), DEFAULT_ELEM_BYTES)
        self.assertEqual(bmm_kernel._l0c_buffering(1), L0C_DOUBLE_BUFFERED)
        self.assertEqual(bmm_kernel._l0c_buffering(L0C_DOUBLE_BUFFERED_BYTES), L0C_DOUBLE_BUFFERED)
        self.assertEqual(bmm_kernel._l0c_buffering(L0C_DOUBLE_BUFFERED_BYTES + 1), L0C_SINGLE_BUFFERED)
        self.assertEqual(bmm_kernel._l0c_buffering(L0C_BYTES), L0C_SINGLE_BUFFERED)

        for n_groups in (1, 2, 8, 27, 28, 29, 30, 56, 100, 1024):
            programs, per_program, ragged = bmm_kernel._split_groups(n_groups)
            self.assertTrue(1 <= programs <= min(n_groups, NUM_CUBE_CORES))
            self.assertEqual(per_program, _cdiv(n_groups, programs))
            if ragged:
                self.assertEqual(programs, min(n_groups, NUM_CUBE_CORES))
            else:
                self.assertEqual(n_groups % programs, 0)

        self.assertAlmostEqual(bmm_kernel._wave_use(NUM_CUBE_CORES), 1.0)
        self.assertAlmostEqual(bmm_kernel._wave_use(2 * NUM_CUBE_CORES), 1.0)
        self.assertAlmostEqual(bmm_kernel._wave_use(1), 1.0 / NUM_CUBE_CORES)
        self.assertEqual(bmm_kernel._window(1), 1)
        self.assertEqual(bmm_kernel._window(1024), round(NUM_CUBE_CORES ** 0.5))
        for m_tiles in (2, 5, 7, 64):
            self.assertTrue(1 <= bmm_kernel._window(m_tiles) <= m_tiles)

    def test_tiling_helpers(self):
        # A 64x64 fp16 tile stages the whole of K=256 in one turn.
        self.assertEqual(bmm_kernel._k_step(64, 64, 256, 2), 256)
        self.assertEqual(bmm_kernel._k_step(64, 64, 100, 2) % MMAD_K_FRACTAL, 0)
        # An extent too large for L0A can stage nothing at all.
        self.assertEqual(bmm_kernel._k_step(1 << 16, 64, 256, 4), 0)
        self.assertEqual(bmm_kernel._k_steps(64, 64, 256, 2), [256])
        self.assertEqual(bmm_kernel._k_steps(1 << 16, 64, 256, 4), [])
        steps = bmm_kernel._k_steps(64, 64, 1024, 2)
        self.assertLessEqual(len(steps), 3)
        self.assertEqual(steps[-1], 512)
        self.assertTrue(all(s % MMAD_K_FRACTAL == 0 and s <= 1024 for s in steps))

        tilings = bmm_kernel._derived_tilings(1024, 1024, 1024, 2)
        self.assertTrue(tilings)
        self.assertLessEqual(len(tilings), 8)
        for block_m, block_n in tilings:
            self.assertEqual(block_m % MMAD_M_FRACTAL, 0)
            self.assertEqual(block_n % MMAD_M_FRACTAL, 0)
            self.assertLessEqual(block_m * block_n * ACCUM_BYTES, L0C_BYTES)
        self.assertTrue(bmm_kernel._wants_single_buffered_tile(256, 256))
        self.assertFalse(bmm_kernel._wants_single_buffered_tile(255, 512))

        for extent in (100, 513, 1024, 4096):
            widths = bmm_kernel._low_waste_widths(extent, 64)
            self.assertTrue(widths)
            for width in widths:
                self.assertEqual(width % MMAD_M_FRACTAL, 0)
                self.assertTrue(16 <= width <= 512)
        for batch, m_tiles, n, block_n in [(1, 1, 100, 64), (1, 2, 512, 256),
                                           (4, 4, 4096, 256), (8, 1, 30, 64)]:
            width = bmm_kernel._rebalanced_n(batch, m_tiles, n, block_n)
            self.assertEqual(width % MMAD_K_FRACTAL, 0)
            self.assertTrue(16 <= width <= block_n)

        # The wide single buffered tile is only offered when it can be filled.
        self.assertIn((256, 256), bmm_kernel._tile_shapes(1024, 1024, 1024, 2))
        self.assertNotIn((256, 256), bmm_kernel._tile_shapes(64, 64, 64, 2))
        self.assertEqual(bmm_kernel._merged_group(1, 1), 16)
        self.assertEqual(bmm_kernel._merged_group(8, 8), 2)
        # An output already one fractal wide gains nothing from folding.
        self.assertEqual(bmm_kernel._merged_group(16, 16), 0)

    def test_config_generators(self):
        m, n, k = 128, 256, 256
        configs = bmm_kernel._get_npu_bmm_configs(m, n, k, elem_bytes=2, batch=4, dtype=None)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertIn(cfg["GROUP_M"], (1, 8))
            self.assertIn(cfg["L1_COPIES"], (0, L1_BUFFER_COPIES))
            self.assertEqual(cfg["EVEN_K"], k % cfg["BLOCK_K"] == 0)
            self.assertEqual(cfg["K_MAIN"], (k // cfg["BLOCK_K"]) * cfg["BLOCK_K"])
            for axis, extent in (("M", m), ("N", n)):
                block = cfg[f"BLOCK_{axis}"]
                rem = extent % block != 0
                self.assertEqual(cfg[f"SHIFT_{axis}"], rem and extent >= block)
                self.assertEqual(cfg[f"MASK_{axis}"], rem and extent < block)
        # Dynamic shapes have no tiling to offer.
        s = sympy.Symbol("s0", integer=True, positive=True)
        self.assertEqual(bmm_kernel._get_npu_bmm_configs(128, 256, s), [])

        batch, m, n, k = 4, 128, 256, 256
        configs = bmm_kernel._get_npu_bmm_tile_configs(batch, m, n, k, elem_bytes=2, dtype=None)
        self.assertTrue(configs)
        for cfg in configs:
            self.assertEqual(cfg["M_TILES"], _cdiv(m, cfg["BLOCK_M"]))
            self.assertEqual(cfg["N_TILES"], _cdiv(n, cfg["BLOCK_N"]))
            self.assertEqual(cfg["TOTAL_TILES"], batch * cfg["M_TILES"] * cfg["N_TILES"])
            self.assertEqual(cfg["WINDOW"], bmm_kernel._window(cfg["M_TILES"]))
        self.assertEqual(bmm_kernel._get_npu_bmm_tile_configs(0, m, n, k, elem_bytes=2), [])

    def test_specialised_config_generators(self):
        # Batches folded into the N direction of one mmad.
        batch, m, n, k = 1024, 8, 8, 64
        configs = bmm_kernel._get_npu_bmm_merged_configs(batch, m, n, k, 2)
        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg["TOTAL_GROUPS"], _cdiv(batch, cfg["G"]))
        self.assertTrue(1 <= cfg["PROGRAMS"] <= NUM_CUBE_CORES)
        self.assertEqual(cfg["GROUPS_PER_PROGRAM"], _cdiv(cfg["TOTAL_GROUPS"], cfg["PROGRAMS"]))
        self.assertEqual(cfg["SHIFT_Q"], batch % cfg["G"] != 0)
        self.assertEqual(cfg["ROWS"] % MMAD_M_FRACTAL, 0)
        self.assertEqual(cfg["COLS"] % MMAD_K_FRACTAL, 0)

        # Batches folded into the M direction of one mmad.
        batch, m, n, k = 1024, 16, 16, 64
        configs = bmm_kernel._get_npu_bmm_batched_configs(batch, m, n, k, 2)
        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertGreaterEqual(cfg["BLOCK_Q"], 2)
        self.assertEqual(cfg["TOTAL_GROUPS"], _cdiv(batch, cfg["BLOCK_Q"]))
        self.assertTrue(1 <= cfg["N_GROUP"] <= NUM_CUBE_CORES)
        self.assertEqual(cfg["GROUPS_PER_PROGRAM"], _cdiv(cfg["TOTAL_GROUPS"], cfg["N_GROUP"]))
        self.assertEqual(cfg["MASKED_M"], cfg["M_TILES"] * cfg["BLOCK_M"] != m)
        self.assertEqual(cfg["MASKED_N"], cfg["N_TILES"] * cfg["BLOCK_N"] != n)
        self.assertEqual(cfg["SHIFT_Q"], batch % cfg["BLOCK_Q"] != 0)
        self.assertEqual(cfg["L1_COPIES"], L1_BUFFER_COPIES)
        self.assertIn(L0C_BUFFERING_KEY, cfg)

        # Too few batches, an unaligned K, a fractal-sized output, an N of one,
        # an L1 overflow and a one-batch group are all declined.
        for shape in [(64, 8, 8, 64), (1024, 8, 8, 63), (1024, 16, 8, 64), (1024, 8, 1, 64)]:
            self.assertEqual(bmm_kernel._get_npu_bmm_merged_configs(*shape, 2), [])
        for shape in [(1, 16, 16, 64), (1024, 16, 16, 63), (1024, 1024, 16, 1024), (100, 16, 16, 64)]:
            self.assertEqual(bmm_kernel._get_npu_bmm_batched_configs(*shape, 2), [])

    def test_dispatch_helpers(self):
        # A batch stride of zero is one matrix broadcast across the batch.
        for size, stride, expected in [((4, 8, 16), (128, 16, 1), True),
                                       ((4, 8, 16), (0, 16, 1), False),
                                       ((8, 16), (16, 1), False)]:
            self.assertEqual(bmm_kernel._carries_batch(_FakeNode(size, stride)), expected)

        s = sympy.Symbol("s0", integer=True, positive=True)
        for size1, size2, expected in [((4, 8, 16), (4, 16, 32), False),
                                       ((4, 1, 16), (4, 16, 32), True),
                                       ((4, 8, 1), (4, 1, 32), True),
                                       ((4, 8, 16), (4, 16, 1), True),
                                       ((4, s, 16), (4, 16, 32), False),
                                       ((8, 16), (16, 32), False)]:
            self.assertEqual(bmm_kernel._degenerate_axis(_FakeNode(size1), _FakeNode(size2)), expected)

        # One tile over the output: only splitting K can feed the cores.
        self.assertTrue(bmm_kernel._only_k_split_fills_cores(1, 256, 128, 8192))
        self.assertFalse(bmm_kernel._only_k_split_fills_cores(1, 256, 128, 8191))
        self.assertFalse(bmm_kernel._only_k_split_fills_cores(NUM_CUBE_CORES, 256, 128, 8192))
        self.assertTrue(bmm_kernel._only_k_split_fills_cores(2, 384, 128, 9000))

        mat1 = _FakeNode((4, 8, 16), (128, 16, 1))
        mat2 = _FakeNode((4, 16, 32), (512, 32, 1))
        layout = _FakeLayout((4, 8, 32), (256, 32, 1))
        zero = _FakeNode((4, 16, 32), (0, 32, 1))
        other = _FakeNode((4, 16, 32), (600, 32, 1))
        self.assertTrue(bmm_kernel.is_batch_stride_largest_or_zero(mat1, mat2, layout))
        # A batch stride of zero is allowed, any other stride is not.
        self.assertTrue(bmm_kernel.is_batch_stride_largest_or_zero(mat1, zero, layout))
        self.assertFalse(bmm_kernel.is_batch_stride_largest_or_zero(mat1, other, layout))

    def test_grids(self):
        meta = {"BLOCK_M": 32, "BLOCK_N": 32}
        for args, expected in [((4, 64, 64), (16, 1, 1)), ((2, 65, 33), (2 * 3 * 2, 1, 1))]:
            self.assertEqual(_call_grid(bmm_kernel.npu_bmm_grid, *args, meta, cdiv=_cdiv), expected)
        for grid_fn, meta in [(bmm_kernel.npu_bmm_tile_grid, {"TOTAL_TILES": 37}),
                              (bmm_kernel.npu_bmm_merged_grid, {"PROGRAMS": 28}),
                              (bmm_kernel.npu_bmm_batched_grid, {"N_GROUP": 7})]:
            expected = next(iter(meta.values()))
            self.assertEqual(_call_grid(grid_fn, 4, 64, 64, meta, cdiv=_cdiv), (expected, 1, 1))


class TestNpuBmmCompile(TestUtils):
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

    def op_bmm(self, mat1, mat2):
        return torch.bmm(mat1, mat2)

    def op_bmm_transposed_b(self, mat1, mat2):
        return torch.bmm(mat1, mat2.transpose(1, 2))

    def op_bmm_relu(self, mat1, mat2):
        return torch.relu(torch.bmm(mat1, mat2))

    def op_baddbmm(self, bias, mat1, mat2):
        return torch.baddbmm(bias, mat1, mat2)

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

    @parametrize("shape", [(2, 64, 64, 64), (4, 128, 64, 256)])
    @parametrize("dtype", ["float32", "float16", "bfloat16"])
    def test_bmm_cases(self, shape, dtype):
        batch, m, k, n = shape
        self._run(self.op_bmm, (self._inputs((batch, m, k), dtype),
                                self._inputs((batch, k, n), dtype)), dtype)

    def test_bmm_transposed_epilogue_baddbmm_and_degenerate(self):
        # A col-major second operand and a fused epilogue reuse the same lowering.
        self._run(self.op_bmm_transposed_b, (self._inputs((2, 64, 64), "float16"),
                                             self._inputs((2, 96, 64), "float16")), "float16")
        self._run(self.op_bmm_relu, (self._inputs((2, 64, 64), "float16"),
                                     self._inputs((2, 64, 64), "float16")), "float16")

        batch, m, k, n = 4, 64, 32, 32
        # K == 1 and N == 1 take the decomposed path instead of a tile.
        for mat2_shape in [(batch, 1, n), (batch, k, 1)]:
            self._run(self.op_bmm, (self._inputs((batch, m, mat2_shape[1]), "float32"),
                                    self._inputs(mat2_shape, "float32")), "float32")
        # A broadcast operand must be declined by the batch specialised templates
        # without losing correctness.
        self._run(self.op_bmm, (self._inputs((batch, m, k), "float32"),
                                self._inputs((1, k, n), "float32").expand(batch, k, n)),
                  "float32")

        batch, m, k, n = 2, 64, 32, 64
        args = (self._inputs((batch, m, n), "float32"),
                self._inputs((batch, m, k), "float32"),
                self._inputs((batch, k, n), "float32"))
        self._run(self.op_baddbmm, args, "float32")

        def op_scaled(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, beta=0.5, alpha=2.0)

        self._run(op_scaled, args, "float32")

    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON",
                  force_disable_caches=True)
    def test_max_autotune_choices_and_declines(self):
        # max_autotune makes tuned_bmm() call add_npu_triton_bmm_choices(), which
        # dispatches to the merged, batched and tiled tiers.
        args = (self._inputs((4, 128, 128), "float16"), self._inputs((4, 128, 128), "float16"))
        offered = []
        orig_offer = bmm_kernel.add_npu_triton_bmm_choices

        def counting_offer(*args, **kwargs):
            offered.append(1)
            return orig_offer(*args, **kwargs)

        with mock.patch.object(bmm_kernel, "add_npu_triton_bmm_choices", counting_offer):
            self._run(self.op_bmm, args, "float16")
        self.assertTrue(offered, "tuned_bmm never offered NPU template choices")

        # A K known only at run time cannot be gated, so the shape is declined and
        # answered by the operator library instead of being guessed.
        compiled = torch.compile(self.op_bmm, backend="inductor", dynamic=True)
        for k in (64, 96):
            mat1 = self._inputs((2, 32, k), "float16")
            mat2 = self._inputs((2, k, 32), "float16")
            self.assertEqual(compiled(mat1, mat2), torch.bmm(mat1, mat2), atol=1e-2, rtol=1e-2)

        # One tile over a long K: only splitting K would fill the cores, which no
        # template does, so the shape is declined as well.
        self._run(self.op_bmm, (self._inputs((1, 8, 8192), "float16"),
                                self._inputs((1, 8192, 8), "float16")), "float16")

    def test_bmm_dynamic_batch(self):
        compiled = torch.compile(self.op_bmm, backend="inductor", dynamic=True)
        for batch in (2, 4):
            mat1 = self._inputs((batch, 64, 32), "float32")
            mat2 = self._inputs((batch, 32, 64), "float32")
            self.assertEqual(compiled(mat1, mat2), torch.bmm(mat1, mat2), atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestNpuBmmKernelHelpers)
instantiate_parametrized_tests(TestNpuBmmCompile)

if __name__ == "__main__":
    run_tests()
