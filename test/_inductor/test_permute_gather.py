# Owner(s): ["module: tests"]
import contextlib
import io

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils

import torch_npu  # noqa: F401
from torch_npu._inductor.triton_experimental import config as ncfg
from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonKernel

# Rewrite markers asserted in the generated kernel source (run_and_get_code):
# gather = "tl.gather" (flat DMA + register permute); trans = "tl.make_block_ptr"
# + "boundary_check" (block_ptr + tl.trans, tails exact). Mode selection and the
# huge-H trans+K-tile path: see config.permute_gather_mode.
GATHER_MARKER = "tl.gather"
TRANS_MARKER = "tl.make_block_ptr"
TRANS_BC_MARKER = "boundary_check"


class TestPermuteGather(TestUtils):

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self._saved_pg = ncfg.enable_permute_gather
        self._saved_pin_xr = ncfg.pin_xr
        # In-process compile: forked workers cannot re-init NPU; threads=1 kills the pool.
        self._saved_threads = torch._inductor.config.compile_threads
        torch._inductor.config.compile_threads = 1

    def tearDown(self):
        torch._inductor.config.compile_threads = self._saved_threads
        ncfg.enable_permute_gather = self._saved_pg
        ncfg.pin_xr = self._saved_pin_xr
        torch._dynamo.reset()
        super().tearDown()

    @staticmethod
    def _t5(arg1, arg0):
        # T5 fused pattern: out[b,k,i] = sum_j (arg1[b,k,i,j] + arg0[i,j,k]^T)
        return (arg1 + arg0.permute(2, 0, 1).unsqueeze(0)).sum(-1)

    def _run(self, h, i=48, j=300, b=2, dynamic=False, enabled=True):
        """Compile the T5 pattern and return (out, codes, eager_ref)."""
        ncfg.enable_permute_gather = enabled
        torch.manual_seed(0)
        arg1 = torch.randn(b, h, i, j, device="npu")
        arg0 = torch.randn(i, j, h, device="npu")
        ref = self._t5(arg1, arg0)
        if dynamic:
            torch._dynamo.mark_dynamic(arg1, 1)
            torch._dynamo.mark_dynamic(arg0, 2)
        fn = torch.compile(
            self._t5,
            options={"npu_backend": "triton_experimental"},
            dynamic=dynamic,
        )
        out, codes = run_and_get_code(fn, arg1, arg0)
        return out, codes, ref

    @parametrize("h", [12, 32])
    def test_gather_mode_small_h(self, h):
        # stride_bytes = h*4 < 256 -> register gather (flat DMA + tl.gather).
        out, codes, ref = self._run(h)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        self.assertIn(GATHER_MARKER, codes[0])
        self.assertNotIn(TRANS_MARKER, codes[0])
        self.assertNotIn("raise ", codes[0])  # P1: no Python raise in any kernel

    @parametrize("h", [64, 72, 1024])
    def test_trans_mode_large_h(self, h):
        # stride_bytes >= 256 -> block_ptr + tl.trans; H=72 (tail 8) / H=1024
        # exercise the int-axis K-tile (tails exact via boundary_check).
        out, codes, ref = self._run(h)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        self.assertIn(TRANS_MARKER, codes[0])
        self.assertIn(TRANS_BC_MARKER, codes[0])
        self.assertNotIn(GATHER_MARKER, codes[0])
        self.assertNotIn("raise ", codes[0])  # P1: no Python raise in any kernel

    @parametrize("h", [72, 100])
    def test_dynamic_h_trans_fallback(self, h):
        # Dynamic (symbolic) H rides trans's shape-generic block_ptr (gather
        # needs a compile-time-affine stride).
        out, codes, ref = self._run(h, dynamic=True)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        self.assertIn(TRANS_MARKER, codes[0])
        self.assertNotIn(GATHER_MARKER, codes[0])

    def test_huge_h_trans_ktile(self):
        # stride_r > max_xblock gates only gather; trans stays available
        # (XBLOCK = min(stride_r, ktile)), so H=8192 rides trans+K-tile.
        out, codes, ref = self._run(8192)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        self.assertIn(TRANS_MARKER, codes[0])
        self.assertIn(TRANS_BC_MARKER, codes[0])
        self.assertNotIn(GATHER_MARKER, codes[0])

    def test_disabled_no_rewrite(self):
        # Flag off (default): no rewrite marker; realize/strided path stays correct.
        out, codes, ref = self._run(72, enabled=False)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        self.assertNotIn(GATHER_MARKER, codes[0])
        self.assertNotIn(TRANS_MARKER, codes[0])

    def _with_slot_hook(self, hook):
        """Run ``hook(geo0)`` on the rewrite's slot sources right before
        _npu_pg_rewrite_body reads them (injects scheduler-produced layouts the
        triton_experimental linearize never emits). Records whether the hook
        fired so an injected test cannot silently degrade to the plain path."""
        orig = NPUTritonKernel._npu_pg_rewrite_body
        self._slot_hook_fired = False

        def wrapped(kernel):
            cands = getattr(kernel, "_npu_pg_candidates", {})
            if cands:
                for geo0 in cands.values():
                    hook(geo0)
                self._slot_hook_fired = True
            return orig(kernel)

        NPUTritonKernel._npu_pg_rewrite_body = wrapped
        return orig

    def test_gather_slot_order_r_first(self):
        # Review: gather pidx is [int, r] flat but reshaped per-slot -- an
        # R-first layout (r_slot < int_slot) used to transpose silently.
        # Force the swap; the rewrite must stay exact for any slot order.
        def swap(geo0):
            vtd = geo0["int_node"].root.var_tensor_dims
            r_tree = geo0["r_tree"]
            int_name = geo0["int_node"].name
            vtd[int_name], r_tree.tensor_dim = r_tree.tensor_dim, vtd[int_name]

        orig = self._with_slot_hook(swap)
        try:
            out, codes, ref = self._run(12)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
            self.assertIn(GATHER_MARKER, codes[0])
        finally:
            NPUTritonKernel._npu_pg_rewrite_body = orig
        self.assertTrue(self._slot_hook_fired)

    def test_missing_slot_falls_back_strided(self):
        # Review: a legitimately-missing slot (vtd key absent / tensor_dim
        # None) used to raise TypeError; it must fall back to strided.
        def drop(geo0):
            geo0["int_node"].root.var_tensor_dims.pop(geo0["int_node"].name, None)

        orig = self._with_slot_hook(drop)
        try:
            out, codes, ref = self._run(12)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
            self.assertNotIn(GATHER_MARKER, codes[0])
        finally:
            NPUTritonKernel._npu_pg_rewrite_body = orig
        self.assertTrue(self._slot_hook_fired)

    def test_pin_xr_does_not_override_pg(self):
        # Review: pin_xr (TEMP DIAGNOSTIC) used to return before the PG pin,
        # compiling the rewritten body under an unvalidated tiling. The PG
        # marker must win. h=48 (unused elsewhere) so the inductor text-hash
        # cache cannot mask the pin.
        ncfg.pin_xr = "256,16"
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                out, codes, ref = self._run(48)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
            self.assertIn(GATHER_MARKER, codes[0])
            # Pre-fix the pin_xr tiling never matched the rewrite's pin ->
            # compile failure -> small-block fallback; the PG pin winning
            # means no such fallback.
            self.assertNotIn("All initial configs failed", stderr.getvalue())
            self.assertNotIn("raise ", codes[0])
        finally:
            ncfg.pin_xr = self._saved_pin_xr


instantiate_parametrized_tests(TestPermuteGather)

if __name__ == "__main__":
    run_tests()
