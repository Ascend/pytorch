# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental must register a native fallback for npu.npu_gather_backward.

The backward graphs of gather / take_along_dim / quantile / nanquantile /
multi_margin_loss carry ``npu.npu_gather_backward.default`` as their only private
op.  It has neither an upstream lowering nor a decomposition, so the bulk pass of
``lowering._register_npu_inductor_fallbacks`` cannot see it and a strict-mode
compile (``implicit_fallbacks=False``) failed with ``MissingOperatorWithoutDecomp``.
These tests pin its registration in ``EXPLICIT_FALLBACK_LIST`` and check that the
compiled backward calls the native op and matches eager gradients.  Two of the
eleven reported cases -- ``multi_margin_loss`` float64 and ``multilabel_margin_loss``
float32 -- are not asserted here; they fail for unrelated root causes.
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
)

aten = torch.ops.aten
npu = torch.ops.npu

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")

GATHER_BACKWARD = npu.npu_gather_backward.default


class TestTeExplicitFallbackList(TestCase):
    """Checks on the list itself and on the wiring that consumes it."""

    def test_gather_backward_listed_for_explicit_fallback(self):
        self.assertIn(GATHER_BACKWARD, lowering_fallback_list.EXPLICIT_FALLBACK_LIST)

    def test_registrar_reads_the_same_list_object(self):
        # A stale import location would register nothing in silence.
        self.assertIs(
            lowering.EXPLICIT_FALLBACK_LIST,
            lowering_fallback_list.EXPLICIT_FALLBACK_LIST,
        )

    def test_entries_are_concrete_op_overloads(self):
        # One concrete overload per line keeps what is registered and what is
        # recorded in FALLBACK_LIST in one-to-one correspondence.
        for op in lowering_fallback_list.EXPLICIT_FALLBACK_LIST:
            self.assertIsInstance(op, torch._ops.OpOverload, op)


class TestNpuGatherBackwardFallback(TestCase):
    def _make_inputs(self, tensors, needs_grad):
        return [
            t.detach().clone().requires_grad_(True) if g else t.detach().clone()
            for t, g in zip(tensors, needs_grad)
        ]

    def _check_backward_matches_eager(self, fn, tensors, needs_grad):
        """Compile ``fn`` under triton_experimental strict mode, run a backward, and
        compare gradients with eager autograd; also check the generated code really
        calls the native op."""
        eager_inputs = self._make_inputs(tensors, needs_grad)
        fn(*eager_inputs).sum().backward()
        eager_grads = {
            i: t.grad.detach().clone()
            for i, (t, g) in enumerate(zip(eager_inputs, needs_grad))
            if g and t.grad is not None
        }

        compiled_inputs = self._make_inputs(tensors, needs_grad)
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
                # Strict mode: without the explicit fallback the compile below raises
                # MissingOperatorWithoutDecomp instead of falling back silently.
                config.patch("implicit_fallbacks", False),
                mock.patch.object(GraphLowering, "save_output_code", save_output_code),
            ):
                torch._dynamo.reset()
                compiled(*compiled_inputs).sum().backward()
        finally:
            torch._dynamo.reset()

        # The compile went through, so the registrar must have made the op fall back.
        self.assertIn(GATHER_BACKWARD, lowerings)
        code = "\n".join(source_codes)
        self.assertIn(f"torch.ops.{GATHER_BACKWARD}(", code)
        for i, grad in eager_grads.items():
            torch.testing.assert_close(
                compiled_inputs[i].grad, grad, rtol=1e-2, atol=1e-3
            )

    @requires_npu
    def test_gather_backward(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.randn(8, 5, device="npu", dtype=dtype)
                index = torch.randint(0, 5, (8, 3), device="npu")
                self._check_backward_matches_eager(
                    lambda a, b: torch.gather(a, 1, b), [x, index], [True, False]
                )

    @requires_npu
    def test_take_along_dim_backward(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.randn(8, 5, device="npu", dtype=dtype)
                index = torch.randint(0, 5, (8, 3), device="npu")
                self._check_backward_matches_eager(
                    lambda a, b: torch.take_along_dim(a, b, 1),
                    [x, index],
                    [True, False],
                )

    @requires_npu
    def test_quantile_backward(self):
        for name, fn in (
            ("quantile", lambda a: torch.quantile(a, 0.5)),
            ("nanquantile", lambda a: torch.nanquantile(a, 0.5)),
        ):
            with self.subTest(fn=name):
                x = torch.randn(4, 6, device="npu")
                self._check_backward_matches_eager(fn, [x], [True])

    @requires_npu
    def test_multi_margin_loss_backward(self):
        x = torch.randn(8, 4, device="npu")
        target = torch.randint(0, 4, (8,), device="npu")
        self._check_backward_matches_eager(
            lambda a, b: aten.multi_margin_loss(a, b, p=1, margin=1.0),
            [x, target],
            [True, False],
        )


if __name__ == "__main__":
    run_tests()
