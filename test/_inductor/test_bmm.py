import sympy
import torch
from torch.testing._internal.common_utils import run_tests
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
    get_num_cube_cores,
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
            self.assertTrue(1 <= programs <= min(n_groups, get_num_cube_cores()))
            self.assertEqual(per_program, _cdiv(n_groups, programs))
            if ragged:
                self.assertEqual(programs, min(n_groups, get_num_cube_cores()))
            else:
                self.assertEqual(n_groups % programs, 0)

        self.assertAlmostEqual(bmm_kernel._wave_use(get_num_cube_cores()), 1.0)
        self.assertAlmostEqual(bmm_kernel._wave_use(2 * get_num_cube_cores()), 1.0)
        self.assertAlmostEqual(bmm_kernel._wave_use(1), 1.0 / get_num_cube_cores())
        self.assertEqual(bmm_kernel._window(1), 1)
        self.assertEqual(bmm_kernel._window(1024), round(get_num_cube_cores() ** 0.5))
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

    @mock.patch.object(bmm_kernel, "get_num_cube_cores", lambda: 28)
    def test_cyclic_bmm_group_coverage(self):
        make_cyclic = bmm_kernel._cyclic_bmm_config
        base = dict(BLOCK_Q=4, BLOCK_M=128, BLOCK_N=64, TOTAL_GROUPS=256,
                    M_TILES=1, N_TILES=1, MASKED_M=False, MASKED_N=False, SHIFT_Q=False)
        original = base.copy()
        for groups in (29, 55, 57, base["TOTAL_GROUPS"]):
            with self.subTest(groups=groups):
                cfg = make_cyclic(dict(base, TOTAL_GROUPS=groups))
                programs, waves = cfg["N_GROUP"], cfg["GROUPS_PER_PROGRAM"]
                self.assertEqual(programs, 28)
                self.assertTrue(cfg["CYCLIC_GROUPS"])
                # Check that the last wave has neither holes nor duplicate writers.
                visited = [wave * programs + pid for wave in range(waves) for pid in range(programs)
                           if wave < groups // programs or pid < groups % programs]
                self.assertEqual(visited, list(range(groups)))
        cyclic = make_cyclic(base)
        self.assertEqual((cyclic["FUSED_BLOCK_M"], cyclic["FUSED_BLOCK_N"]),
                         (base["BLOCK_M"], base["BLOCK_N"] // 2))
        self.assertEqual(base, original)

        for changes in (dict(TOTAL_GROUPS=28), dict(TOTAL_GROUPS=56), dict(SHIFT_Q=True),
                        dict(MASKED_M=True), dict(MASKED_N=True), dict(M_TILES=2), dict(N_TILES=2),
                        dict(BLOCK_N=48)):
            with self.subTest(changes=changes):
                self.assertIsNone(make_cyclic(dict(base, **changes)))
        max_groups = (2**31 - 1) // base["BLOCK_Q"]
        self.assertIsNotNone(make_cyclic(dict(base, TOTAL_GROUPS=max_groups)))
        self.assertIsNone(make_cyclic(dict(base, TOTAL_GROUPS=max_groups + 1)))

    def test_bmm_l2_bypass_requires_full_output_and_aligned_rows(self):
        streams = bmm_kernel._bmm_streams_past_l2
        capacity = 128 * 1024**2
        self.assertTrue(streams(2048, 256, 128, 64, 2, capacity))
        # Both inputs plus C exceed L2 even at the smaller batch.
        self.assertFalse(streams(2046, 256, 128, 64, 2, capacity))
        self.assertFalse(streams(2048, 256, 128, 64, 2, capacity + 1))
        for n, k, l2_bytes in ((129, 64, capacity), (128, 65, capacity), (128, 64, 0)):
            with self.subTest(n=n, k=k, l2_bytes=l2_bytes):
                self.assertFalse(streams(2048, 256, n, k, 2, l2_bytes))

    def test_specialised_config_generators(self):
        # Batches folded into the N direction of one mmad.
        batch, m, n, k = 1024, 8, 8, 64
        configs = bmm_kernel._get_npu_bmm_merged_configs(batch, m, n, k, 2)
        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg["TOTAL_GROUPS"], _cdiv(batch, cfg["G"]))
        self.assertTrue(1 <= cfg["PROGRAMS"] <= get_num_cube_cores())
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
        self.assertTrue(1 <= cfg["N_GROUP"] <= get_num_cube_cores())
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
        self.assertFalse(bmm_kernel._only_k_split_fills_cores(get_num_cube_cores(), 256, 128, 8192))
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


if __name__ == "__main__":
    run_tests()
