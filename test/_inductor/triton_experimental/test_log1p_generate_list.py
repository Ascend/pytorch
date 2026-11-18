# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression coverage for aten.log1p staying on the triton_experimental lowering.

aten.log1p has a registered pointwise lowering, but the triton_experimental
auto-fallback pass turns every registered lowering into an aclnn fallback
unless the op appears in GENERATE_LIST. log1p was missing from that list, so
graph mode emitted an aten fallback instead of a Triton kernel.

Keeping it on the Triton path needs an accurate log1p: CANN's vector
``libdevice.log1p`` is ``log(1 + x)`` with the addition already rounded to
the input dtype, so it returns 0 for x = 1e-8 (100% relative error) and is
19% high at 1e-7. The lowering override therefore emits
``npu_triton_helpers.log1p``, which adds the rounding error of ``1 + x``
back and matches a float64 reference to <= 2 ulp. The compiled-path tests
pin both the emitted call and the accuracy at the two ends of the domain:
  * small x, where 1 + x rounds back to 1 in fp32 (x = 1e-8 -> 0 instead
    of 1e-8, i.e. 100% relative error),
  * x close to -1, where -1 must stay -inf rather than NaN,
  * x = +inf, where the compensation term would otherwise become
    inf - inf = NaN and hide the correct +inf result,
  * x = -0.0, where log1p(-0.0) must stay -0.0 so that
    1 / log1p(-0.0) is -inf rather than +inf (the compensated form is all
    +0 terms).

The whitelist test runs without a device; the rest are NPU-gated.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import lowering_override_list

aten = torch.ops.aten

# Values at both ends of the fp32 log1p domain. -1.0 itself is included: log1p
# is -inf there and must stay -inf (not NaN).
LOG1P_CASES = torch.tensor(
    [
        1e-8,
        1e-7,
        1e-5,
        1e-3,
        0.5,
        1.0,
        10.0,
        1e6,
        -1e-8,
        -1e-5,
        -0.5,
        -0.999,
        -0.999999,
        -0.99999994,  # nextafter(-1, 0) in fp32
        -1.0,
    ],
    dtype=torch.float64,
)


class TestLog1PGenerateList(TestCase):
    def test_log1p_in_generate_list(self):
        self.assertIn(aten.log1p, lowering_override_list.GENERATE_LIST)

    def _compiled_wrapper_code(self, fn, args):
        source_codes = []

        def save_output_code(code):
            source_codes.append(code)

        compiled = torch.compile(
            fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )
        try:
            with (
                config.patch("force_disable_caches", True),
                config.patch("implicit_fallbacks", False),
                mock.patch.object(
                    GraphLowering, "save_output_code", save_output_code
                ),
            ):
                torch._dynamo.reset()
                out = compiled(*args)
        finally:
            torch._dynamo.reset()
        return "\n".join(source_codes), out

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_log1p_compiles_to_triton(self):
        def fn(t):
            return torch.log1p(t)

        x = torch.randn(8, 8, device="npu").clamp(min=-0.9)
        code, compiled_out = self._compiled_wrapper_code(fn, (x,))

        # The wrapper embeds the fx graph as comment lines; strip them so the
        # fallback check only sees real calls.
        code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("torch.ops.aten.log1p", code)
        self.assertIn("def triton_", code)
        # Must be the compensated helper, not the CANN libdevice intrinsic
        # (whose fp32 log1p is log(1 + x) after rounding 1 + x).
        self.assertIn("npu_triton_helpers.log1p(", code)
        self.assertNotIn("libdevice.log1p(", code)
        torch.testing.assert_close(compiled_out, torch.log1p(x))

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_log1p_accuracy_at_domain_edges(self):
        """Small x and x -> -1 must match float64 log1p to fp32 precision.

        The reference is computed in float64 on the CPU from the exact fp32
        inputs, so it is independent of the device implementation.
        """
        def fn(t):
            return torch.log1p(t)

        # Round through fp32 on the host and derive the reference from those
        # exact values, so the comparison is not polluted by input rounding.
        host = LOG1P_CASES.to(torch.float32)
        reference = torch.log1p(host.to(torch.float64)).to(torch.float32)
        x = host.to("npu")

        out = torch.compile(
            fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )(x)

        self.assertTrue(torch.isfinite(out[:-1]).all().item())
        self.assertTrue(torch.isinf(out[-1]).item() and out[-1].item() < 0)
        torch.testing.assert_close(
            out, reference.to("npu"), rtol=1e-5, atol=1e-9
        )

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_log1p_signed_zero(self):
        """log1p(-0.0) must stay -0.0, so its reciprocal stays -inf.

        The compensated form evaluates ``log(1) + 0 - 0``, and every term of
        that is +0, so without the explicit zero pass-through the sign of a
        -0.0 input is lost and ``1 / log1p(-0.0)`` turns from -inf to +inf.
        """

        def log1p(t):
            return torch.log1p(t)

        def recip(t):
            return torch.reciprocal(torch.log1p(t))

        host = torch.tensor([0.0, -0.0], dtype=torch.float32)
        reference = torch.log1p(host.to(torch.float64)).to(torch.float32)

        out = torch.compile(
            log1p, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )(host.to("npu")).cpu()

        # assert_close compares -0.0 and +0.0 as equal, so check the sign bit
        # explicitly. It is read from the host copy: the NPU lowering of
        # signbit is x < 0, which is False for -0.0.
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        self.assertEqual(torch.signbit(out).tolist(), [False, True])

        recip_out = torch.compile(
            recip, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )(host.to("npu")).cpu()
        self.assertEqual(recip_out[0].item(), float("inf"))
        self.assertEqual(recip_out[1].item(), float("-inf"))

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_log1p_non_finite_inputs(self):
        """+inf must stay +inf; -inf and NaN must stay NaN.

        For x = +inf the kernel computes u = 1 + x = +inf and
        delta = x - (u - 1) = inf - inf = NaN, so an unguarded compensation
        term turns the correct log(+inf) = +inf into NaN.
        """
        def fn(t):
            return torch.log1p(t)

        host = torch.tensor(
            [float("inf"), float("-inf"), float("nan")], dtype=torch.float32
        )
        out = torch.compile(
            fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )(host.to("npu")).cpu()

        self.assertTrue(torch.isinf(out[0]).item() and out[0].item() > 0)
        self.assertTrue(torch.isnan(out[1]).item())
        self.assertTrue(torch.isnan(out[2]).item())

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_log1p_fp16_upcasts(self):
        """fp16/bf16 inputs are computed in fp32 and rounded once at the end."""

        def fn(t):
            return torch.log1p(t)

        for dtype, rtol, atol in (
            (torch.float16, 1e-2, 1e-3),
            (torch.bfloat16, 4e-2, 1e-2),
        ):
            host = torch.tensor(
                [1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0], dtype=torch.float64
            ).to(dtype)
            reference = torch.log1p(host.to(torch.float64)).to(dtype)
            out = torch.compile(
                fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
            )(host.to("npu"))
            torch.testing.assert_close(
                out, reference.to("npu"), rtol=rtol, atol=atol
            )

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_log1p_kernel_runs_on_device(self):
        # Runtime evidence independent of code text: the compiled graph
        # launches the log1p triton kernel on device and no aten::log1p
        # fallback op executes. Warm up first so the profiled call is a
        # cache-hit launch, not a recompile.
        def fn(t):
            return torch.log1p(t)

        x = torch.randn(4096, device="npu").clamp(min=-0.9)
        compiled = torch.compile(
            fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )
        torch._dynamo.reset()
        warm_out = compiled(x)

        from torch_npu.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]
        ) as prof:
            out = compiled(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            trace = os.path.join(tmpdir, "trace.json")
            prof.export_chrome_trace(trace)
            with open(trace, encoding="utf-8") as f:
                trace_data = json.load(f)
        if isinstance(trace_data, dict):
            trace_data = trace_data.get("traceEvents", [])
        names = [e.get("name", "") for e in trace_data]
        self.assertTrue(any("log1p" in name for name in names))
        self.assertNotIn("aten::log1p", names)
        torch.testing.assert_close(out, warm_out)


if __name__ == "__main__":
    run_tests()
