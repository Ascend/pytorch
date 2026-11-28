"""Host-only contracts for user-defined triton kernels on the NPU backend.

Nothing here imports torch_npu, allocates a device tensor or runs Inductor: every
regression pinned below is visible in the source itself.

A user-defined triton kernel reaches Inductor as the triton_kernel_wrapper_*
higher-order ops, whose upstream lowering builds ir.UserDefinedTritonKernel and
therefore the launch (and the kernel's side effects). The triton_experimental
fallback sweep used to replace the mutation variant with a generic aten
fallback, which dropped the launch silently: wrong results, no error. The fix
lives in two places that must stay in sync, and both are pinned here.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "torch_npu/_inductor/triton_experimental/lowering.py"
FALLBACK_NAMES = ROOT / "torch_npu/_inductor/triton_experimental/lowering_fallback_list.py"

HOPS = (
    "triton_kernel_wrapper_mutation",
    "triton_kernel_wrapper_functional",
)


class TestUserTritonKernelContracts(unittest.TestCase):
    def test_sweep_keeps_the_hop_out_of_the_fallback(self):
        # The guard must run before make_fallback(op): after it, the upstream
        # lowering has already been clobbered and the launch is gone.
        src = SWEEP.read_text(encoding="utf-8")
        self.assertIn("keep_upstream", src)
        self.assertIn("if op in keep_upstream:", src)
        keep = src.index("if op in keep_upstream:")
        self.assertLess(keep, src.index("make_fallback(op)"))

    def test_keep_set_resolves_both_variants_defensively(self):
        # This runs during backend activation, so a torch build that does not
        # expose one variant must not raise AttributeError there -- that would
        # turn "silently wrong results" into "backend cannot be registered".
        compact = "".join(SWEEP.read_text(encoding="utf-8").split())
        for hop in HOPS:
            self.assertIn(f'getattr(_tkw,"{hop}",None)', compact)

    def test_hop_is_not_in_the_explicit_fallback_names(self):
        # The sweep also force-fallbacks every name in this list, so removing the
        # HOP from the sweep guard alone would not be enough.
        src = FALLBACK_NAMES.read_text(encoding="utf-8")
        for hop in HOPS:
            self.assertNotIn(f'"higher_order.{hop}"', src)


if __name__ == "__main__":
    unittest.main()
