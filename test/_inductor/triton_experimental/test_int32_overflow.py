# Owner(s): ["module: tests"]
import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    instantiate_parametrized_tests,
    TestCase,
)

import torch_npu  # noqa: F401


# Regression test for the int32 index-overflow bug in the triton_experimental
# NPU backend (2026-08).
#
# Background: for a reduction kernel whose total element count
# (numel * reduction_numel) exceeds int32_max, select_index_dtype() selects
# int64 indexing. Before the fix, that signal was propagated to every index
# construction (pid / arange tile / scalar odometer), which (a) let the
# overflowing pointer addend (e.g. 268435456*x1) wrap to a negative address in
# int32 → Ascend vector-core fault 507035, and (b) when the whole kernel was
# upcast to int64 instead, doubled UB on big arange tiles → vector-core timeout
# 507034. The fix keeps the big arange tiles int32 and widens EVERY axis
# factor in the address expressions at the load/store use site
# (coeff*var.to(tl.int64)) — unconditional, constructive correctness.
#
# The >2^31-scale end-to-end cases (17-18 GiB HBM each) were removed on
# 2026-09-16; recover them from git history if the boundary needs re-checking.


class TestInt32Overflow(TestCase):

    def test_mask_cmp_lhs_int64_narrow_protects_fp32(self):
        # With mask_cmp_fp32 on, the int32 narrow must
        # stay TERMINAL — falling through to the fp32 compare hangs the
        # kernel (triton-ascend lowers the fp32 mask compare into vector-core
        # work that never completes: 507034 vector-core timeout on a minimal
        # non-reduction kernel, >10 min no result on pointwise; verified
        # 2026-08-14). Pin the generated mask LHS texts: the default-off
        # layout byte-for-byte, and the fp32-on layout keeping the int64-narrow
        # case on the runnable int32 compare. These are pure-text assertions
        # (no compile) so they stay green while the fp32 path is broken.
        import torch_npu._inductor.triton_experimental.codegen.npu_header as nh
        import sympy

        off_cases = [
            # (label, index_expr, numel, index_dtype, expected_lhs)
            ("int64-narrow", "x0index", 256, "tl.int64", "(x0index).to(tl.int32)"),
            ("int64-at-boundary", "x0index", 2**31, "tl.int64", "x0index"),
            ("int64-dynamic", "x0index", sympy.Symbol("s"), "tl.int64", "x0index"),
            ("int32", "x0index", 256, "tl.int32", "x0index"),
        ]
        orig = nh.npu_mask_cmp_fp32
        try:
            nh.npu_mask_cmp_fp32 = False
            for label, e, n, d, want in off_cases:
                self.assertEqual(
                    nh._mask_cmp_lhs(e, n, d), want,
                    f"mask_cmp_fp32 off: {label}",
                )
            # fp32 on: the int64 narrow keeps the int32 compare (protection);
            # the in-range int32 case is unchanged (the pre-fix fp32 cast).
            nh.npu_mask_cmp_fp32 = True
            self.assertEqual(
                nh._mask_cmp_lhs("x0index", 256, "tl.int64"),
                "(x0index).to(tl.int32)",
                "int64 narrow must not fall through to the hanging fp32 compare",
            )
            self.assertEqual(
                nh._mask_cmp_lhs("x0index", 256, "tl.int32"),
                "(x0index).to(tl.float32)",
            )
        finally:
            nh.npu_mask_cmp_fp32 = orig

    def test_non_linearize_in_range_reduction_grid_exact(self):
        # In-range non-linearize reduction: the launcher grid must be exactly
        # ceil(xnumel/XBLOCK). The heuristics' grid_0 defaults to the persistent
        # NPU_CU_COUNT (48) for reduction kernels, which is only valid under the
        # linearize structure's group dispatch — a non-linearize kernel
        # (xoffset = pid*XBLOCK, always-true xmask, no group folding) over-reads
        # past the input when 48 > ceil (MTE fault 507035, the pre-fix failure)
        # and silently drops tiles when 48 < ceil (uninitialized output rows, the
        # min(ceil, 48) clamp regression). Codegen flags
        # inductor_meta["npu_linearize"]=False; the heuristics then launches the
        # exact tile count instead of the fixed 48.
        import torch_npu._inductor.triton_experimental.codegen.triton as tmod

        orig = tmod.triton_codegen_linearize
        tmod.triton_codegen_linearize = False
        try:
            x = torch.randn(64, 128, 256, device="npu")
            ref = x.sum(dim=-1)

            def fn(t):
                return t.sum(dim=-1)

            cf = torch.compile(fn, options={"npu_backend": "triton_experimental"})
            y, codes = run_and_get_code(cf, x)
            torch.npu.synchronize()

            # Numeric correctness pins both failure modes: the 507035 over-read
            # (pre-fix) and the missing-tile clamp regression (wrong results).
            self.assertTrue(torch.allclose(y, ref, atol=1e-3, rtol=1e-3))
            # Classic non-linearize structure, in-range so no int64 promotion.
            self.assertIn("xoffset = tl.program_id(0) * XBLOCK", codes[0])
            self.assertNotIn("tl.int64", codes[0])
        finally:
            tmod.triton_codegen_linearize = orig



    def test_expand_over_int32_blocks_odometer_i64(self):
        # Oversized-block-count dispatch audit: a static numel >= 2^31 must
        # never become
        # a triton literal. Three poison paths found and fixed by the
        # expand->sum >2^31 probe: (1) codegen_static_numels stomping the i64
        # runtime arg with a bare literal (uint32 typing in [2^31, 2^32) ->
        # signedness errors in the div/mod dispatch chains), (2) per-axis
        # tl.constexpr numels in the linearize header, (3) the r-tree
        # tt.equal_to constants specialization (uint32->i64 vcast rejected by
        # BiShengIR). All three keep the numel on its i64 runtime arg, so the
        # whole block-dispatch scalar chain promotes to int64. This case has
        # > 2^31 total blocks via a stride-0 expand axis (numel 2.2e9,
        # storage ONE element) — cheap, no HBM skip needed.
        n = 2_200_000_000  # > 2^31, inside the uint32 window [2^31, 2^32)
        x = torch.ones((1,), device="npu").expand((n,))
        ref = x.sum()

        def fn(t):
            return t.sum()

        cf = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        y, codes = run_and_get_code(cf, x)
        torch.npu.synchronize()
        self.assertTrue(torch.allclose(y, ref, rtol=1e-5))
        # no bare >= 2^31 numel literal in any generated kernel (stomp or
        # constexpr forms)
        for code in codes:
            self.assertNotRegex(code, r"numel = 2\d{9}")
            self.assertNotRegex(code, r"numel : tl.constexpr = 2\d{9}")

    def test_dtype_role_isolation_pins(self):
        # P0 pin: the dtype-role separation is currently BY CONVENTION (three
        # mutually-counteracting global patches; see NPUTritonKernel.
        # dtype_to_str's SCOPE comment for the audited truth). These pins turn
        # silent drift (upstream sync, future edits) into loud failures.
        import torch._inductor.utils as inductor_utils

        import torch_npu._inductor.triton_experimental.codegen.triton as te_triton

        # (a) in-range kernel stays byte-clean int32: the widening must not
        # leak into kernels that do not need it
        x = torch.randn(1024, 64, device="npu")

        cf = torch.compile(lambda t: t.sum(1), options={"npu_backend": "triton_experimental"})
        y, codes = run_and_get_code(cf, x)
        self.assertTrue(torch.allclose(y, x.sum(1), atol=1e-3, rtol=1e-3))
        self.assertNotIn("tl.int64", codes[0])

        # (b) compute types are NOT demoted: npu_triton_compute_type has no
        # int64 branch and bypasses the demotion mapping
        self.assertEqual(te_triton.npu_triton_compute_type(torch.int64), "tl.int64")

        # (c) seam policy pins: the mapping itself still demotes bare
        # triton_type calls (kept for default-backend coexistence), but the
        # two former seams are explicitly OPEN — int64 stores stay tl.int64
        # (npu_triton_store_type) and non-int64 dtypes delegate upstream.
        # Any change in either direction must be a reviewed decision.
        # Requires activation (the compile above installs the patches).
        import torch._inductor.codegen.triton as up_codegen

        self.assertEqual(inductor_utils.triton_type(torch.int64), "tl.int32")
        self.assertEqual(up_codegen.triton_store_type(torch.int64), "tl.int64")
        self.assertEqual(up_codegen.triton_store_type(torch.bool), "tl.int8")
        self.assertEqual(
            up_codegen.triton_store_type(torch.float32), "tl.float32"
        )


instantiate_parametrized_tests(TestInt32Overflow)

if __name__ == "__main__":
    run_tests()
