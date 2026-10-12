# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental must register native fallbacks for nearest{1,3}d upsample backward.

TE's ``disable_implicit_decomposition`` pops the forward nearest-exact py_kernels, so
the grad graph carries the bare ``aten.upsample_nearest{1,3}d_backward`` op.  The 2d
variant has an upstream lowering and passes; the 1d/3d variants have none and were
missing from ``EXPLICIT_FALLBACK_LIST``, so a TE-compiled backward raised
``MissingOperatorWithoutDecomp`` under ``implicit_fallbacks=False`` while the default
backend passed.  These tests pin both ops in the list and check the compiled gradient
matches eager.  Fixes #4934.
"""

import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import lowering_fallback_list

aten = torch.ops.aten

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")


class TestTeNearestBackwardFallback(TestCase):
    def _check_backward_matches_eager(self, fn, input):
        """Compile ``fn`` under triton_experimental strict mode, run a backward, and
        compare the gradient with eager autograd."""
        eager_input = input.detach().clone().requires_grad_(True)
        fn(eager_input).sum().backward()
        eager_grad = eager_input.grad.detach().clone()

        compiled = torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        compiled_input = input.detach().clone().requires_grad_(True)
        try:
            with (
                config.patch("force_disable_caches", True),
                # Strict mode: without the explicit fallback the compile below raises
                # MissingOperatorWithoutDecomp instead of falling back silently.
                config.patch("implicit_fallbacks", False),
            ):
                torch._dynamo.reset()
                loaded_backend = (
                    torch_npu.utils._dynamo._InductorNpuRegistry._loaded_backend
                )
                if loaded_backend != "triton_experimental":
                    raise AssertionError(
                        "triton_experimental backend did not take effect for this compile"
                    )
                compiled(compiled_input).sum().backward()
        finally:
            torch._dynamo.reset()

        torch.testing.assert_close(
            compiled_input.grad, eager_grad, rtol=1e-2, atol=1e-3
        )

    def test_nearest_backward_listed_for_explicit_fallback(self):
        for op in (
            aten.upsample_nearest1d_backward.default,
            aten.upsample_nearest3d_backward.default,
        ):
            self.assertIn(op, lowering_fallback_list.EXPLICIT_FALLBACK_LIST)

    @requires_npu
    def test_upsample_nearest1d_backward_fallback(self):
        def fn(a):
            return F.interpolate(a, size=(16,), mode="nearest")

        self._check_backward_matches_eager(fn, torch.randn(1, 3, 8, device="npu"))

    @requires_npu
    def test_upsample_nearest3d_backward_fallback(self):
        def fn(a):
            return F.interpolate(a, size=(4, 8, 8), mode="nearest")

        self._check_backward_matches_eager(
            fn, torch.randn(1, 3, 2, 4, 4, device="npu")
        )


if __name__ == "__main__":
    run_tests()
