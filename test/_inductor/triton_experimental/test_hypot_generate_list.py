"""Regression coverage for hypot staying on the triton_experimental lowering.

aten.hypot has a registered pointwise lowering (torch_npu/_inductor/
lowering_fx.py), but the triton_experimental auto-fallback pass makes every
registered lowering a fallback unless its op appears in GENERATE_LIST. Before
the fix hypot was missing from that list, so graph mode silently emitted an
aten fallback instead of the libdevice.hypot Triton kernel and upcast
counters saw 0 upcasts (issue #5002).

The whitelist membership test runs without a device and guards the
GENERATE_LIST entry; the compiled-path tests prove the wrapper compiles a
triton kernel calling libdevice.hypot with no aten.hypot fallback call, and
a profiler test captures the kernel launch on device as runtime evidence.
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


class TestHypotGenerateList(TestCase):
    def test_hypot_in_generate_list(self):
        self.assertIn(aten.hypot, lowering_override_list.GENERATE_LIST)

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

    def _check_hypot_compiled(self, dtype):
        def fn(a, b):
            return torch.hypot(a, b)

        a = torch.randn(8, 8, device="npu", dtype=dtype)
        b = torch.randn(8, 8, device="npu", dtype=dtype)
        eager_out = fn(a, b)

        code, compiled_out = self._compiled_wrapper_code(fn, (a, b))

        # The wrapper embeds the fx graph as comment lines; strip them so the
        # fallback check only sees real calls.
        code = "\n".join(
            line for line in code.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("torch.ops.aten.hypot", code)
        self.assertIn("async_compile.triton(", code)
        self.assertIn("libdevice.hypot(", code)
        torch.testing.assert_close(compiled_out, eager_out)

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_hypot_compiles_to_triton_kernel_fp32(self):
        self._check_hypot_compiled(torch.float32)

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_hypot_compiles_to_triton_kernel_fp16(self):
        self._check_hypot_compiled(torch.float16)

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_hypot_kernel_runs_on_device(self):
        # Runtime evidence independent of code text: the compiled graph
        # launches the hypot triton kernel on device and no aten::hypot
        # fallback op executes. Warm up first so the profiled call is a
        # cache-hit launch, not a recompile.
        def fn(a, b):
            return torch.hypot(a, b)

        a = torch.randn(8, 8, device="npu")
        b = torch.randn(8, 8, device="npu")
        compiled = torch.compile(
            fn, fullgraph=True, options={"npu_backend": "triton_experimental"}
        )
        torch._dynamo.reset()
        warm_out = compiled(a, b)

        from torch_npu.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]
        ) as prof:
            out = compiled(a, b)

        # This torch_npu build's profile object exposes only trace export,
        # so read the exported chrome trace for event names. Depending on
        # the build the JSON is either {"traceEvents": [...]} or a bare list.
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = os.path.join(tmpdir, "trace.json")
            prof.export_chrome_trace(trace)
            with open(trace, encoding="utf-8") as f:
                trace_data = json.load(f)
        if isinstance(trace_data, dict):
            trace_data = trace_data.get("traceEvents", [])
        names = [e.get("name", "") for e in trace_data]
        self.assertTrue(any("fused_hypot" in name for name in names))
        self.assertNotIn("aten::hypot", names)
        torch.testing.assert_close(out, warm_out)


if __name__ == "__main__":
    run_tests()
