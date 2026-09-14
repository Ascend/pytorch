import os
import unittest
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import lowering_override_list


class TestSpatialNativeFallback(TestCase):
    def test_spatial_ops_use_explicit_fallback_list(self):
        expected = (
            torch.ops.aten.reflection_pad2d.default,
            torch.ops.aten.reflection_pad2d_backward.default,
            torch.ops.aten.upsample_bilinear2d.default,
            torch.ops.aten.upsample_bilinear2d_backward.default,
        )
        for op in expected:
            self.assertIn(op, lowering_override_list.EXPLICIT_FALLBACK_LIST)

    def _check_native_forward_backward(self, fn, input, native_ops):
        eager_input = input.detach().clone().requires_grad_()
        compiled_input = input.detach().clone().requires_grad_()

        eager_output = fn(eager_input)
        grad_output = torch.randn_like(eager_output)
        eager_output.backward(grad_output)

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
                compiled_output = compiled(compiled_input)
                compiled_output.backward(grad_output)
        finally:
            torch._dynamo.reset()

        self.assertEqual(eager_output, compiled_output)
        self.assertEqual(eager_input.grad, compiled_input.grad)
        code = "\n".join(source_codes)
        for native_op in native_ops:
            self.assertIn(f"torch.ops.aten.{native_op}.default(", code)

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_reflection_pad2d_native_forward_backward(self):
        def fn(input):
            return F.pad(input, (1, 2, 1, 2), mode="reflect")

        self._check_native_forward_backward(
            fn,
            torch.randn(2, 3, 5, 6, device="npu"),
            ("reflection_pad2d", "reflection_pad2d_backward"),
        )

    def _check_bilinear_upsample(self, **kwargs):
        def fn(input):
            return F.interpolate(
                input, mode="bilinear", align_corners=False, **kwargs
            )

        self._check_native_forward_backward(
            fn,
            torch.randn(2, 3, 5, 7, device="npu"),
            ("upsample_bilinear2d", "upsample_bilinear2d_backward"),
        )

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_bilinear_upsample_size_native_forward_backward(self):
        self._check_bilinear_upsample(size=(9, 11))

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_bilinear_upsample_scale_factor_native_forward_backward(self):
        self._check_bilinear_upsample(scale_factor=(1.5, 2.0))


if __name__ == "__main__":
    run_tests()
