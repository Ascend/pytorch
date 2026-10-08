"""Device gate of the triton_experimental native-dropout decomposition override.

Regression guard for #4939: ``_override_native_dropout_decomp()`` rewrites the
``aten.native_dropout`` / ``aten.native_dropout_backward`` entries of torch's
inductor decomposition tables.  Before the fix the rewrite applied to *every*
device, so importing ``torch_npu._inductor`` diverted pure CPU dropout graphs
onto private kernels (``npu._npu_dropout``) that CPU lowering cannot compile,
raising ``MissingOperatorWithoutDecomp``.

The pinned contract is: NPU inputs keep the npu-kernel decomposition; inputs
on any other device fall back to the vanilla-torch decomposition captured
before the override (aten-only targets), i.e. exactly what would have run
without torch_npu imported -- in both random modes: with fast random
(``config.fallback_random=False``) the merged ``select_decomp_table()``
delegates to the captured vanilla RNG decomposition, and with
``fallback_random=True`` the inductor base table -- which never carried a
forward ``aten.native_dropout`` entry in vanilla torch -- returns
``NotImplemented``, leaving the node for the FallbackKernel path.
"""

import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx
from torch._inductor.decomposition import decompositions as ind_decomps
from torch._inductor.decomposition import select_decomp_table
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._python_dispatch import TorchDispatchMode

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

aten = torch.ops.aten
NATIVE_DROPOUT = aten.native_dropout.default
NATIVE_DROPOUT_BACKWARD = aten.native_dropout_backward.default
PRIVATE_NPU_TARGETS = ("npu._npu_dropout", "npu.npu_dropout_backward")


class _OpRecorder(TorchDispatchMode):
    """Record every dispatched operator while the block runs."""

    def __init__(self):
        super().__init__()
        self.targets = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.targets.append(str(func))
        return func(*args, **(kwargs or {}))


def _private_npu_targets(targets):
    return [t for t in targets if any(p in t for p in PRIVATE_NPU_TARGETS)]


_OVERRIDE_QUALNAMES = (
    "native_dropout",
    "native_dropout_backward",
    "native_dropout_fast_random",
)


def _is_torch_npu_override(entry):
    # Primary signal: the _npu_decomp_override marker (module moves/renames
    # can't stale it).  Secondary __module__/__qualname__ match covers
    # pre-fix builds with no marker, so the guard fails there instead of
    # skipping quietly.
    if getattr(entry, "_npu_decomp_override", False):
        return True
    return (
        getattr(entry, "__module__", "") == "torch_npu._inductor.decomposition"
        and getattr(entry, "__qualname__", "").rsplit(".", 1)[-1] in _OVERRIDE_QUALNAMES
    )


_OVERRIDES_INSTALLED = (
    _is_torch_npu_override(ind_decomps.get(NATIVE_DROPOUT))
    and _is_torch_npu_override(ind_decomps.get(NATIVE_DROPOUT_BACKWARD))
    and _is_torch_npu_override(extra_random_decomps.get(NATIVE_DROPOUT))
)


@unittest.skipUnless(
    _OVERRIDES_INSTALLED,
    "triton_experimental native-dropout decomposition override is not "
    "installed in this configuration (backend not triton_experimental, or "
    "triton-ascend missing so the backend loader bailed out)",
)
class TestNativeDropoutDecompDeviceGate(TestCase):
    def test_every_cpu_or_meta_input_falls_back_to_vanilla_decomp(self):
        """Gate contract: with the override installed in all three tables, no
        non-NPU input may ever reach a private npu kernel target."""
        probes = (
            ("inductor_table forward", NATIVE_DROPOUT, ind_decomps, "cpu"),
            ("inductor_table backward", NATIVE_DROPOUT_BACKWARD, ind_decomps, "cpu"),
            ("extra_random_decomps forward", NATIVE_DROPOUT, extra_random_decomps, "cpu"),
            ("selected table forward", NATIVE_DROPOUT, select_decomp_table(), "cpu"),
            ("inductor_table forward meta", NATIVE_DROPOUT, ind_decomps, "meta"),
            ("extra_random_decomps forward meta", NATIVE_DROPOUT, extra_random_decomps, "meta"),
            ("selected table backward", NATIVE_DROPOUT_BACKWARD, select_decomp_table(), "cpu"),
            ("inductor_table backward meta", NATIVE_DROPOUT_BACKWARD, ind_decomps, "meta"),
        )
        for name, op, table, device in probes:
            with self.subTest(name):
                entry = table[op]
                if "backward" in name:
                    grad = torch.randn(4, 4, device=device)
                    mask = torch.ones(4, 4, dtype=torch.bool, device=device)
                    with _OpRecorder() as recorder:
                        entry(grad, mask, 2.0)
                else:
                    x = torch.randn(4, 4, device=device)
                    with _OpRecorder() as recorder:
                        entry(x, 0.5, True)
                self.assertEqual(_private_npu_targets(recorder.targets), [])

    def test_fallback_random_mode_forward_gate_returns_not_implemented(self):
        """The subtlest leg of the two-mode contract: with
        ``config.fallback_random=True``, ``select_decomp_table()`` is the
        inductor base table, whose vanilla-torch entry for
        ``aten.native_dropout`` does not exist (the forward decomposition is
        an RNG one living only in ``extra_random_decomps``).  The gated
        wrapper must return ``NotImplemented`` for non-NPU inputs -- leaving
        the node for the FallbackKernel path, exactly as without torch_npu --
        and must emit no private npu target; compiled end-to-end in this
        mode the artifact likewise stays free of npu targets."""
        for device in ("cpu", "meta"):
            with self.subTest(f"fwd gate on {device} returns NotImplemented"):
                with inductor_config.patch("fallback_random", True):
                    entry = select_decomp_table()[NATIVE_DROPOUT]
                    self.assertTrue(_is_torch_npu_override(entry))
                    x = torch.randn(4, 4, device=device)
                    with _OpRecorder() as recorder:
                        result = entry(x, 0.5, True)
                    self.assertIs(result, NotImplemented)
                    self.assertEqual(_private_npu_targets(recorder.targets), [])

        with self.subTest("fallback_random compiled artifact has no npu targets"):
            def fn(x):
                return F.dropout(x, 0.5, True)

            x = torch.randn(8, 8)
            gm, _ = dynamo.export(fn)(x)
            with inductor_config.patch("fallback_random", True):
                compiled = compile_fx(gm, (x,), decompositions=select_decomp_table())
                with _OpRecorder() as recorder:
                    out = compiled(x.clone())
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(_private_npu_targets(recorder.targets), [])

    def test_inductor_table_application_keeps_cpu_dropout_free_of_npu_targets(self):
        """Nightly symptom (#4939): when the decomposition pass consults
        select_decomp_table() for a CPU ``native_dropout`` node, the
        decomposition artifact must contain no private npu targets.  Before
        the fix this died in CPU lowering with MissingOperatorWithoutDecomp
        on target ``npu._npu_dropout.default``."""

        def fn(x):
            return F.dropout(x, 0.5, True)

        x = torch.randn(8, 8)
        gm, _ = dynamo.export(fn)(x)
        compiled = compile_fx(gm, (x,), decompositions=select_decomp_table())
        with _OpRecorder() as recorder:
            out = compiled(x.clone())
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(_private_npu_targets(recorder.targets), [])

    def test_cpu_native_dropout_compiles_end_to_end_without_npu_lowering_gap(self):
        """Smoke check: plain CPU torch.compile(backend="inductor") of
        dropout keeps working after torch_npu._inductor is imported.  This
        leg is deliberately *not* relied on as a regression guard: for a
        graph this simple the pre-dispatch composite decomposition of
        aten.native_dropout can satisfy the compile without the inductor
        table ever being consulted, so this smoke also passed before the
        fix.  The teeth are in the table-level probes and the
        explicit-table compile_fx tests above."""

        def fn(x):
            return F.dropout(x, 0.5, True)

        compiled = torch.compile(fn, backend="inductor")
        x = torch.randn(8, 8)
        with _OpRecorder() as recorder:
            out = compiled(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(_private_npu_targets(recorder.targets), [])

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_npu_inputs_keep_the_npu_kernel_decomposition(self):
        """Behaviour on NPU is unchanged: the override still routes forward
        through npu._npu_dropout and a packed mask backward through
        npu.npu_dropout_backward."""
        x = torch.randn(4, 4, device="npu")

        with _OpRecorder() as recorder:
            _, mask = select_decomp_table()[NATIVE_DROPOUT](x, 0.5, True)
        self.assertIn("npu._npu_dropout", " ".join(recorder.targets))

        grad = torch.randn(4, 4, device="npu")
        self.assertNotEqual(tuple(mask.shape), tuple(grad.shape))  # packed mask
        with _OpRecorder() as recorder:
            ind_decomps[NATIVE_DROPOUT_BACKWARD](grad, mask, 2.0)
        self.assertIn("npu.npu_dropout_backward", " ".join(recorder.targets))


if __name__ == "__main__":
    run_tests()
