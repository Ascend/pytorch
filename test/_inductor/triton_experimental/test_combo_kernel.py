import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import config as te_config


def pointwise_pair(x, y):
    return x.relu(), y.sigmoid()


class TestComboTileAllocation(TestCase):
    def test_core_ranges_cover_each_tile_once(self):
        for cores in (40, 48):
            for tiles in (0, 1, 2, cores - 1, cores, cores + 1, 2 * cores + 1):
                covered = []
                q, r = divmod(tiles, cores)
                for core in range(cores):
                    count = q + (core < r)
                    base = core * q + min(core, r)
                    covered.extend(range(base, base + count))
                self.assertEqual(covered, list(range(tiles)))

    def test_member_prefixes_with_empty_members(self):
        counts = (0, 3, 0, 5)
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        assigned = []
        for tile in range(offsets[-1]):
            member = next(i for i, end in enumerate(offsets[1:]) if tile < end)
            assigned.append((member, tile - offsets[member]))
        self.assertEqual(assigned, [(1, i) for i in range(3)] + [(3, i) for i in range(5)])


@unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
class TestTEComboKernel(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def compile_function(self, fn, *args, enabled=True, dynamic=False):
        with (
            inductor_config.patch(combo_kernels=True),
            te_config.patch(enable_te_combo_kernel=enabled),
        ):
            compiled = torch.compile(
                fn,
                dynamic=dynamic,
                options={"npu_backend": "triton_experimental"},
            )
            result, codes = run_and_get_code(compiled, *args)
        return result, "\n".join(codes)

    def compile_pair(self, x, y, enabled=True):
        return self.compile_function(pointwise_pair, x, y, enabled=enabled)

    def assert_combo_result(self, fn, *args, expected_combo):
        # Keep an owned copy: the NPU allocator may reuse an eager output
        # buffer while the compiled variant is being materialized.
        eager = tuple(value.clone() for value in fn(*args))
        separate, separate_code = self.compile_function(fn, *args, enabled=False)
        # Materialize before the second compilation can recycle NPU buffers.
        separate = tuple(value.clone() for value in separate)
        actual, code = self.compile_function(fn, *args)
        actual = tuple(value.clone() for value in actual)
        torch.testing.assert_close(separate, eager)
        torch.testing.assert_close(actual, eager)
        self.assertNotIn("te_combo_meta", separate_code)
        self.assertEqual("te_combo_meta" in code, expected_combo)
        return code

    def test_single_launch_and_suffixed_headers(self):
        x = torch.randn(10001, device="npu")
        y = torch.randn(10001, device="npu")
        expected = pointwise_pair(x, y)
        actual, code = self.compile_pair(x, y)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want)
        self.assertEqual(code.count("@npu_triton_heuristics.foreach"), 1)
        self.assertEqual(code.count(".run("), 1)
        self.assertIn("te_combo_meta", code)
        for slot in (0, 1):
            for stem in (rf"x\d+_{slot}numel", rf"real_block_x\d+_{slot}", rf"x\d+_{slot}_blocks"):
                self.assertRegex(code, rf"\b{stem}\s*:\s*tl\.constexpr\s*=")
        self.assertNotIn("@triton_heuristics.foreach", code)

    def test_disabled_and_mixed_dtype_fall_back(self):
        x = torch.randn(1025, device="npu")
        y = torch.randn(1025, device="npu")
        for enabled, second in ((False, y), (True, y.half())):
            with self.subTest(enabled=enabled, dtype=second.dtype):
                actual, code = self.compile_pair(x, second, enabled=enabled)
                for got, want in zip(actual, pointwise_pair(x, second)):
                    torch.testing.assert_close(got, want)
                self.assertNotIn("te_combo_meta", code)
                self.assertEqual(code.count(".run("), 2)

    def test_pointwise_ops_and_dtypes(self):
        def fn(x, y):
            return x + 3, y * 2

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32):
            with self.subTest(dtype=dtype):
                if dtype == torch.int32:
                    x = torch.randint(-10, 10, (1025,), device="npu", dtype=dtype)
                    y = torch.randint(-10, 10, (1025,), device="npu", dtype=dtype)
                else:
                    x = torch.randn(1025, device="npu", dtype=dtype)
                    y = torch.randn(1025, device="npu", dtype=dtype)
                code = self.assert_combo_result(fn, x, y, expected_combo=True)
                self.assertEqual(code.count(".run("), 1)

    def test_tile_boundaries(self):
        for size in (0, 1, 255, 256, 257, 10001):
            with self.subTest(size=size):
                x = torch.randn(size, device="npu")
                y = torch.randn(size, device="npu")
                code = self.assert_combo_result(
                    pointwise_pair, x, y, expected_combo=size > 0
                )
                self.assertEqual(code.count(".run("), 0 if size == 0 else 1)

        x = torch.randn((), device="npu")
        y = torch.randn((), device="npu")
        code = self.assert_combo_result(pointwise_pair, x, y, expected_combo=True)
        self.assertEqual(code.count(".run("), 1)
        self.assertIn("'combo_tile_counts': [1, 1]", code)

    def test_three_members_and_shared_readonly_input(self):
        def fn(x, y, z):
            return x.relu(), y.sigmoid(), z.tanh()

        inputs = tuple(torch.randn(1025, device="npu") for _ in range(3))
        code = self.assert_combo_result(fn, *inputs, expected_combo=True)
        self.assertEqual(code.count(".run("), 1)

        def shared(x):
            return x.relu(), x.sigmoid()

        # The scheduler fuses these two consumers into one ordinary pointwise
        # node before ComboKernel admission, so there is no combo candidate.
        code = self.assert_combo_result(shared, inputs[0], expected_combo=False)
        self.assertEqual(code.count(".run("), 1)

    def test_foreach_add_and_unary(self):
        def binary(a, b, c, d):
            return torch._foreach_add([a, b], [c, d])

        def unary(a, b):
            return torch._foreach_abs([a, b])

        inputs = tuple(torch.randn(1025, device="npu") for _ in range(4))
        # TE phase one admits pointwise scheduler nodes only.  Foreach nodes
        # remain correct through the regular fallback path until a dedicated
        # foreach lowering is enabled.
        for fn, fn_inputs in ((binary, inputs), (unary, inputs[:2])):
            code = self.assert_combo_result(fn, *fn_inputs, expected_combo=False)
            self.assertNotIn("npu_triton_heuristics.foreach", code)

    def test_mixed_shapes_and_noncontiguous_fall_back(self):
        x = torch.randn(1025, device="npu")
        y = torch.randn(513, device="npu")
        self.assert_combo_result(pointwise_pair, x, y, expected_combo=False)

        a = torch.randn(16, 16, device="npu").t()
        b = torch.randn(16, 16, device="npu").t()
        self.assert_combo_result(pointwise_pair, a, b, expected_combo=False)

    def test_int64_falls_back(self):
        x = torch.randint(-10, 10, (1025,), device="npu", dtype=torch.int64)
        y = torch.randint(-10, 10, (1025,), device="npu", dtype=torch.int64)
        code = self.assert_combo_result(pointwise_pair, x, y, expected_combo=False)
        self.assertEqual(code.count(".run("), 2)

    def test_member_limit_falls_back(self):
        def three_members(x, y, z):
            return x.relu(), y.sigmoid(), z.tanh()

        inputs = tuple(torch.randn(1025, device="npu") for _ in range(3))
        with te_config.patch(te_combo_max_members=2):
            code = self.assert_combo_result(
                three_members, *inputs, expected_combo=False
            )
        self.assertEqual(code.count(".run("), 3)

    def test_combo_switch_restores_standalone_codegen(self):
        x = torch.randn(1025, device="npu")
        y = torch.randn(1025, device="npu")
        for enabled in (False, True, False):
            with self.subTest(enabled=enabled):
                torch._dynamo.reset()
                actual, code = self.compile_pair(x, y, enabled=enabled)
                torch.testing.assert_close(actual, pointwise_pair(x, y))
                self.assertEqual("te_combo_meta" in code, enabled)
                self.assertEqual(code.count(".run("), 1 if enabled else 2)

    def test_reduction_and_dynamic_shapes_fall_back(self):
        def reduction(x, y):
            return x.sum(dim=1), y.relu()

        x = torch.randn(32, 16, device="npu")
        y = torch.randn(32, 16, device="npu")
        self.assert_combo_result(reduction, x, y, expected_combo=False)

        a = torch.randn(1025, device="npu")
        b = torch.randn(1025, device="npu")
        eager = tuple(value.clone() for value in pointwise_pair(a, b))
        actual, code = self.compile_function(pointwise_pair, a, b, dynamic=True)
        torch.testing.assert_close(actual, eager)
        self.assertNotIn("te_combo_meta", code)


if __name__ == "__main__":
    run_tests()
