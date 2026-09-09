# Owner(s): ["module: tests"]
import functools

import torch
import torch._dynamo as dynamo
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    instantiate_parametrized_tests,
)
from testutils import TestUtils

import torch_npu  # noqa: F401
import torch_npu._inductor.triton_experimental.config as ncfg


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
# This test pins the behavior at the 2^31 boundary: 8388609*256 == 2^31 + 257
# elements, just past int32_max, so the kernel must go int64-indexed without a
# full tile upcast.
def _free_hbm_bytes():
    try:
        free, _total = torch.npu.mem_get_info()
        return free
    except Exception:
        return None


def skipIfInsufficientHBM(min_free_bytes):
    """Skip a large test when the device's free HBM is verifiably below the
    requirement. The overflow corpus keeps the input, an eager reference,
    outputs and compile caches resident at once, so budget ~2x the tensor."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self):
            free = _free_hbm_bytes()
            if free is not None and free < min_free_bytes:
                self.skipTest(
                    f"large test: needs >= {min_free_bytes >> 30} GiB free "
                    f"HBM, only {free >> 30} GiB available"
                )
            return fn(self)
        return wrapper
    return deco


def _cut_autotune_runs():
    """Test-speedup: cut the mspti autotune benchmark to 1 warmup + 1 active run.

    Guard-value note: the assertions of this suite are codegen-text and
    numeric-correctness checks, only ONE real kernel run is needed; the
    production 25-run-per-candidate bench ranks configs for performance,
    which the tests do not measure. The candidate set itself is left
    untouched (the with_index kernel at > 2^31 elements needs the
    production set: single-candidate pinning was tried and the largest
    tile fails UB compile while small-tile fallbacks produce wrong
    indices). Returns (old_warmup, old_active) for a try/finally restore.
    """
    old = (ncfg.mspti_warmup, ncfg.mspti_active)
    ncfg.mspti_warmup = 1
    ncfg.mspti_active = 1
    return old


# P0 drift pin (2026-08-28 audit): the audited call surface of the type-name
# helpers in the PINNED torch's codegen, keyed by source text without line
# numbers — a moved line does not false-alarm, an added/changed call site
# does. A torch bump failing test_int64_type_callsite_surface_pin means the
# int64 type-role assumptions need a re-audit (update this list only after
# re-auditing).
_EXPECTED_INT64_TYPE_CALLSURFACE = [
    "simd: return self.dtype_to_str(self.get_index_dtype_as_torch_dtype())",
    "triton: asm_triton_type = triton_type(dtype)",
    "triton: cast_inputs.append(f\"{inp}.to({triton_type(input_dtypes[i])})\")",
    "triton: f\"({str(logical_index)}).to({self.dtype_to_str(index_dtype)})\"",
    "triton: f\"({var}).to({triton_type(dtype)})\",",
    "triton: f\".to({triton_type(result_dtype)})\"",
    "triton: f\"{result} = {result}.to({triton_compute_type(target_dtype)})\"",
    "triton: f\"{torch.iinfo(index_dtype).max}, {self.dtype_to_str(index_dtype)})\"",
    "triton: f\"{value}.to({triton_compute_type(dtype)})\",",
    "triton: f\"{value}.to({triton_store_type(store_dtype)})\",",
    "triton: line += f\".to({triton_type(dtype)})\"",
    "triton: out = f\"{out}.to({triton_type(out_dtype)})\"",
    "triton: out = f\"{out}.to({triton_type(upcast_compute_type(dtype))})\"",
    "triton: out = f\"{x}.to({triton_type(dtype)}, bitcast=True)\"",
    "triton: out_dtype = triton_compute_type(dtype)",
    "triton: out_dtype = triton_store_type(dtype)",
    "triton: result = f\"{result}.to({self.dtype_to_str(result_type)})\"",
    "triton: result = f\"{result}.to({triton_type(result_dtype)})\"",
    "triton: return f\"({arg}).to({triton_type(dtype)})\"",
    "triton: return f\"{result}.to({triton_type(dtype)})\"",
    "triton: return f\"{result}.to({triton_type(result_dtype)})\"",
    "triton: return triton_compute_type(upcast_acc_dtype(dtype))",
    "triton: return triton_type(dtype)",
    "triton: return triton_type(upcast_compute_type(dtype))",
    "triton: triton_type = triton_compute_type(dtype)",
    "triton: value = f\"{value}.to({triton_store_type(store_dtype)})\"",
    "triton: x = f\"{x}.to({triton_type(src_dtype)})\"",
    "triton: {result_var}_ws = ({ws_name} + {self.index_to_str(ws_offset)}).to(tl.pointer_type({triton_type(dtype)}))",
]


class TestTritonExperimentalInt32Overflow(TestUtils):

    @skipIfInsufficientHBM(17 * 2**30)
    def test_sum_over_int32_max_promotes_overflow_addend(self):
        # 2^31 + 257 elements (8.0 GiB fp32) — the minimal > int32_max case.
        x = torch.randn(8388609, 256, device=torch.device("npu"))
        ref = x.sum(dim=1)

        def fn(t):
            return t.sum(dim=1)

        cf = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        y, codes = run_and_get_code(cf, x)

        # Numeric correctness at > 2^31 scale.
        self.assertTrue(torch.allclose(y, ref, atol=1e-3, rtol=1e-3))
        # Every axis factor of the address expression is widened to int64 at
        # the load site (unconditional); lanes/tiles stay int32.
        self.assertIn("to(tl.int64)", codes[0])

    @skipIfInsufficientHBM(18 * 2**30)
    def test_dynamic_over_int32_max_unconditional_widen(self):
        # Dynamic axis length whose trace-time hint already exceeds int32_max.
        # The widening is unconditional (every axis factor in the address
        # expression), so there is no snapshot-derived promote decision and no
        # runtime guard contract: correctness holds for ANY runtime shape and
        # the hint boundary is irrelevant. This case used to exercise the
        # guard-skip path; it now pins that the dynamic >2^31 kernel simply
        # compiles and runs correctly.
        s0 = 2_150_000_000  # numel s0*2 = 4.3e9 > 2^31 (8.6 GiB fp16)
        x = torch.full((s0, 2), 1.0, device="npu", dtype=torch.float16)
        dynamo.mark_dynamic(x, 0)
        ref = x.sum(dim=1)

        def fn(t):
            return t.sum(dim=1)

        cf = torch.compile(fn, options={"npu_backend": "triton_experimental"})
        y = cf(x)
        torch.npu.synchronize()

        self.assertTrue(torch.allclose(y, ref))

    @skipIfInsufficientHBM(17 * 2**30)
    def test_non_linearize_over_int32_max_keeps_tiles_int32(self):
        # With codegen_linearize=False, past 2^31 elements the
        # non-linearize structure cannot stay correct — xoffset =
        # pid.to(int64)*XBLOCK upcasts every arange tile to int64 via mixed
        # broadcast (UB doubling, 507034) and the results go wrong. The kernel
        # must force the linearize structure (i64 rides the scalar
        # group_base/real_block chain and the widened address factors, tiles
        # stay int32) regardless of the config, which then only governs
        # in-range kernels.
        import torch_npu._inductor.triton_experimental.codegen.triton as tmod

        orig = tmod.triton_codegen_linearize
        tmod.triton_codegen_linearize = False
        try:
            x = torch.full((8388609, 256), 1.0, device="npu", dtype=torch.float32)
            ref = x.sum(dim=1)

            def fn(t):
                return t.sum(dim=1)

            cf = torch.compile(fn, options={"npu_backend": "triton_experimental"})
            y, codes = run_and_get_code(cf, x)
            torch.npu.synchronize()

            # Numeric correctness at > 2^31 scale with linearize forced off.
            self.assertTrue(torch.allclose(y, ref, atol=1e-3, rtol=1e-3))
            # Reverse-guard the whole-tile upcast: no arange/full tile may be
            # int64 AT ALL (as a dtype literal or via a cast). The pre-fix
            # all-tile upcast emitted tl.arange(..., tl.int64) with no ".to"
            # call, so asserting only on the cast would not catch it (507034).
            for line in codes[0].splitlines():
                if "tl.arange" in line or "tl.full" in line:
                    self.assertNotIn(
                        "tl.int64", line,
                        f"non-linearize tile upcast past 2^31: {line.strip()}",
                    )
            # Forward-guard: the overflowing addend — 256*x0
            # has term_max 256*8388608 == 2^31 > int32_max — must carry the
            # int64 cast. Widening is unconditional (constructive correctness),
            # so other axis factors of the same index may be cast too;
            # exclusivity is deliberately NOT pinned — a wrong widening
            # selection can only cost speed, never correctness.
            cast_lines = [l for l in codes[0].splitlines() if ".to(tl.int64)" in l]
            self.assertTrue(
                cast_lines,
                "no int64 cast in the address expressions at all",
            )
            self.assertRegex(
                "\n".join(cast_lines),
                r"\b256\*x0\.to\(tl\.int64\)",
                f"overflow addend cast missing: {cast_lines}",
            )
        finally:
            tmod.triton_codegen_linearize = orig

    @skipIfInsufficientHBM(17 * 2**30)
    def test_max_with_index_over_int32_max_correct(self):
        # The upstream arg-reduction index accumulator follows
        # select_index_dtype() and is emitted as a full int64 tile past 2^31
        # elements (tl.full(..., tl.int64) + *_with_index compare/select over
        # the whole [X, R] tile). Variant C keeps this upstream default — the
        # tile works on NPU, verified correct — and promotes ONLY the pointer
        # addend to int64 at the load use site. Note aten.argmax itself does
        # NOT reach this path on NPU: it falls back to eager (no triton kernel
        # is generated), so torch.max(dim=1) is the reachable form of the
        # arg-reduction code path.
        #
        # Test time budget: the kernel is > 2^31 logical elements (hard
        # requirement of the overflow guard) and the with_index accumulator is
        # a full int64 tile, so a single real run costs ~3s on AIV; the
        # production autotune would then benchmark 20+ candidates x 25 runs
        # (~15min) without adding guard value. The assertions below need only
        # ONE real run (codegen texts + numeric correctness are
        # config-independent), so cut the mspti bench to 1+1 and leave the
        # candidate set untouched.
        orig_warmup, orig_active = _cut_autotune_runs()
        try:
            x = torch.randn(8388609, 256, device="npu")
            ref_v, ref_i = x.max(dim=1)

            def fn(t):
                return t.max(dim=1)

            cf = torch.compile(fn, options={"npu_backend": "triton_experimental"})
            result, codes = run_and_get_code(cf, x)
            yv, yi = result
            torch.npu.synchronize()

            self.assertTrue(torch.allclose(yv, ref_v, atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.equal(yi, ref_i))
            # The kernel must exist (compiled, not eager) and carry the overflow
            # addend cast (variant C at the load site).
            self.assertIn("with_index", codes[0])
            self.assertIn(".to(tl.int64)", codes[0])
            # R7-c trap pin: the with_index accumulator type and its
            # torch.iinfo(index_dtype).max sentinel fill must stay PAIRED —
            # narrowing dtype_to_str alone would emit the int64 max into a
            # tl.int32 full and break compilation.
            self.assertIn("9223372036854775807, tl.int64", codes[0])
        finally:
            ncfg.mspti_warmup = orig_warmup
            ncfg.mspti_active = orig_active

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

    def test_int64_type_callsite_surface_pin(self):
        # P0 pin: upstream drift detector, paired with
        # _EXPECTED_INT64_TYPE_CALLSURFACE above. Failing on a torch bump is
        # BY DESIGN: re-audit the int64 type-role surface, then update the
        # list.
        import inspect
        import re

        import torch._inductor.codegen.simd as up_simd
        import torch._inductor.codegen.triton as up_triton

        pat = re.compile(r"(dtype_to_str|triton_type|triton_compute_type|triton_store_type)\(")
        surface = set()
        for mod in (up_triton, up_simd):
            short = mod.__name__.rsplit(".", 1)[-1]
            for line in open(inspect.getsourcefile(mod)).read().splitlines():
                s = line.strip()
                if pat.search(s) and not s.startswith(("def ", "#")):
                    surface.add(f"{short}: {s}")
        self.assertEqual(
            sorted(surface),
            sorted(_EXPECTED_INT64_TYPE_CALLSURFACE),
            "int64 type-helper call surface drifted — re-audit the dtype "
            "roles before updating this snapshot",
        )


instantiate_parametrized_tests(TestTritonExperimentalInt32Overflow)

if __name__ == "__main__":
    run_tests()
