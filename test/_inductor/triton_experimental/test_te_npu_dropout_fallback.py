# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental must register native fallbacks for the dropout ops.

The native_dropout decomposition routes the forward through
``npu._npu_dropout.default`` (which returns a packed uint8 mask) and, once that
falls back, the autograd chain surfaces ``npu.npu_dropout_backward.default``.
Neither op has a triton_experimental lowering, so a strict-mode compile
(``implicit_fallbacks=False``) failed with ``MissingOperatorWithoutDecomp``.
These tests pin both registrations and check that a compiled forward+backward
goes through the native ops and matches eager gradients.  The three reported
CPU-variant cases are not asserted here: their green depends on the
device-gating in the native_dropout decomposition (#4939), not on this list.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch._inductor.lowering import lowerings
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import (
    lowering,
    lowering_fallback_list,
    lowering_override_list,
)

npu = torch.ops.npu

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")

DROPOUT = npu._npu_dropout.default
DROPOUT_BACKWARD = npu.npu_dropout_backward.default


class TestTeDropoutExplicitFallbackList(TestCase):
    """Checks on the dropout registrations and the wiring that consumes them."""

    def test_dropout_ops_listed_for_explicit_fallback(self):
        self.assertIn(DROPOUT, lowering_fallback_list.EXPLICIT_FALLBACK_LIST)
        self.assertIn(DROPOUT_BACKWARD, lowering_fallback_list.EXPLICIT_FALLBACK_LIST)

    def test_dropout_not_also_in_generate_list(self):
        # _npu_dropout used to sit in GENERATE_LIST, which only exempts an op
        # from the blanket fallback; with no TE lowering that left it ownerless
        # and the compile still failed. It must now live only in the explicit
        # list (one home per op).
        self.assertNotIn(npu._npu_dropout, lowering_override_list.GENERATE_LIST)

    def test_registrar_reads_the_same_list_object(self):
        # A stale import location would register nothing in silence.
        self.assertIs(
            lowering.EXPLICIT_FALLBACK_LIST,
            lowering_fallback_list.EXPLICIT_FALLBACK_LIST,
        )


class TestNpuDropoutFallback(TestCase):
    def _dropout(self, x):
        return torch.nn.functional.dropout(x, p=0.3, training=True)

    def _compiled_matches_eager(self, fn, tensor):
        """Compile ``fn`` under triton_experimental strict mode, run a backward,
        and compare the input gradient with eager autograd. Both sides are
        seeded to the same value first so the philox mask is identical; without
        the explicit fallback the compile below raises
        MissingOperatorWithoutDecomp instead of falling back silently."""
        eager = tensor.detach().clone().requires_grad_(True)
        torch.manual_seed(0)
        fn(eager).sum().backward()
        eager_grad = eager.grad.detach().clone()

        compiled_in = tensor.detach().clone().requires_grad_(True)
        compiled = torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        source_codes = []

        def save_output_code(code):
            source_codes.append(code)

        try:
            with (
                config.patch("force_disable_caches", True),
                config.patch("implicit_fallbacks", False),
                mock.patch.object(GraphLowering, "save_output_code", save_output_code),
            ):
                torch._dynamo.reset()
                torch.manual_seed(0)
                compiled(compiled_in).sum().backward()
        finally:
            torch._dynamo.reset()

        self.assertIn(DROPOUT, lowerings)
        code = "\n".join(source_codes)
        self.assertIn(f"torch.ops.{DROPOUT}(", code)
        torch.testing.assert_close(
            compiled_in.grad, eager_grad, rtol=1e-2, atol=1e-3
        )

    @requires_npu
    def test_dropout_forward_backward(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.randn(16, 8, device="npu", dtype=dtype)
                self._compiled_matches_eager(self._dropout, x)

    @requires_npu
    def test_dropout_via_linear(self):
        # nn.Dropout inside a module: exercises the same private op through a
        # realistic training graph (dropout on the activations, then a loss).
        def fn(x):
            dropped = torch.nn.functional.dropout(x, p=0.4, training=True)
            return (dropped * 2.0).tanh()

        x = torch.randn(16, 16, device="npu")
        self._compiled_matches_eager(fn, x)


if __name__ == "__main__":
    run_tests()
