# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental must register a native fallback for grid_sampler_3d.

``aten.grid_sampler_3d`` has neither an upstream inductor lowering nor a
decomposition -- on CUDA it compiles only because ``implicit_fallbacks`` is on by
default -- so the bulk fallback-registration pass cannot see it and a strict-mode
triton_experimental compile fails with ``MissingOperatorWithoutDecomp``, which is
how the three ``test_comprehensive_grid_sampler_3d_npu_*`` opinfo cases tripped.
The NPU op-plugin provides aclnnGridSampler3D for both directions, so the forward
and backward overloads go into ``EXPLICIT_FALLBACK_LIST``.  These tests pin the
registration and check that a strict-mode compile runs the native op and matches
eager outputs and gradients for the three failing dtypes.
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

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")

FORWARD = aten.grid_sampler_3d.default
FORWARD_OUT = aten.grid_sampler_3d.out
BACKWARD = aten.grid_sampler_3d_backward.default
BACKWARD_OUT = aten.grid_sampler_3d_backward.out

ALL_OVERLOADS = (FORWARD, FORWARD_OUT, BACKWARD, BACKWARD_OUT)


class TestTeGridSampler3dExplicitFallbackList(TestCase):
    """Device-independent checks on the registration and the wiring that consumes it."""

    def test_grid_sampler_3d_listed_for_explicit_fallback(self):
        for op in ALL_OVERLOADS:
            self.assertIn(op, lowering_fallback_list.EXPLICIT_FALLBACK_LIST)

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

    def test_all_four_overloads_registered_as_fallbacks(self):
        # The loader ran at import time (TE backend selected above); the
        # registrar must have moved every listed overload into ``lowerings``
        # and recorded it in ``FALLBACK_LIST``.  This covers the two ``.out``
        # overloads, which no opinfo graph exercises on its own.
        for op in ALL_OVERLOADS:
            self.assertIn(op, lowerings)
            self.assertIn(op, lowering.FALLBACK_LIST)


class TestNpuGridSampler3dFallback(TestCase):
    @requires_npu
    def test_grid_sampler_3d_native_forward_backward(self):
        def fn(input, grid):
            return torch.nn.functional.grid_sample(
                input,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        for dtype, rtol, atol in (
            (torch.float16, 2e-2, 2e-3),
            (torch.float32, 1e-4, 1e-5),
            (torch.float64, 1e-5, 1e-6),
        ):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                # opinfo comprehensive shape for grid_sampler_3d, grad on both
                # tensor inputs so the joint backward runs the backward op too.
                input = torch.rand(
                    2, 3, 4, 5, 6, device="npu", dtype=dtype, requires_grad=True
                )
                grid = torch.rand(
                    2, 3, 4, 5, 3, device="npu", dtype=dtype, requires_grad=True
                )

                eager_output = fn(input, grid)
                eager_output.sum().backward()
                eager_grads = (input.grad.detach().clone(), grid.grad.detach().clone())
                input.grad = None
                grid.grad = None

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
                        # Strict mode: without the explicit fallbacks the compile
                        # raises MissingOperatorWithoutDecomp instead of falling
                        # back silently the way CUDA's default config does.
                        config.patch("implicit_fallbacks", False),
                        mock.patch.object(
                            GraphLowering, "save_output_code", save_output_code
                        ),
                    ):
                        torch._dynamo.reset()
                        compiled_output = compiled(input, grid)
                        compiled_output.sum().backward()
                finally:
                    torch._dynamo.reset()

                code = "\n".join(source_codes)
                self.assertIn(f"torch.ops.{FORWARD}(", code)
                self.assertIn(f"torch.ops.{BACKWARD}(", code)
                torch.testing.assert_close(
                    compiled_output, eager_output, rtol=rtol, atol=atol
                )
                torch.testing.assert_close(
                    input.grad, eager_grads[0], rtol=rtol, atol=atol
                )
                torch.testing.assert_close(
                    grid.grad, eager_grads[1], rtol=rtol, atol=atol
                )


if __name__ == "__main__":
    run_tests()
