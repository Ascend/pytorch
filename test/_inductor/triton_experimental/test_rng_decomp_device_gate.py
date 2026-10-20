# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

"""TE backend: RNG decomps must device-gate philox away from NPU (issue 5249).

``_npu_rng_gate`` wraps bernoulli / uniform / rrelu_with_noise decomps to
return NotImplemented on npu inputs, so the aten op reaches an eager fallback
(bit-exact) instead of philox codegen, which cannot replay the NPU RNG stream.
Non-npu inputs keep the original decomposition; table tests need no device.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import pytest
import torch
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401

aten = torch.ops.aten

requires_npu = pytest.mark.skipif(
    not torch.npu.is_available(), reason="requires an NPU device"
)

# Packets device-gated by the TE registration hook.  Iterating overloads()
# keeps the guard honest if upstream adds a new overload.
_GATED_PACKETS = [
    aten.bernoulli,
    aten.uniform,
    aten.uniform_,
    aten.rrelu_with_noise,
    aten.rrelu_with_noise_,
    aten.rrelu_with_noise_functional,
]


class TestRngDecompDeviceGate(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _merged_table(self):
        # select_decomp_table() merges decompositions with extra_random_decomps
        # and memoizes; go through it so both source tables are covered.
        return select_decomp_table()

    def test_rng_decomps_are_device_gated(self):
        table = self._merged_table()
        checked = 0
        for packet in _GATED_PACKETS:
            for overload in packet.overloads():
                op = getattr(packet, overload)
                fn = table.get(op)
                if fn is None:
                    continue
                checked += 1
                self.assertTrue(
                    getattr(fn, "_npu_rng_gate", False),
                    f"{op} present on the TE path but not device-gated -- "
                    "philox would functionalize it and desync the NPU RNG stream",
                )
        self.assertGreater(checked, 0, "no RNG decomp found to gate (table empty?)")

    def test_native_dropout_override_not_gated(self):
        # The pruning must not touch the npu private dropout decomposition
        # installed by _override_native_dropout_decomp().
        table = self._merged_table()
        fn = table.get(aten.native_dropout.default)
        self.assertIsNotNone(fn, "native_dropout override missing")
        self.assertFalse(
            getattr(fn, "_npu_rng_gate", False),
            "native_dropout must keep its private npu kernel, not be philox-gated",
        )

    @requires_npu
    def test_gate_shortcircuits_npu_and_passes_cpu(self):
        table = self._merged_table()
        fn = table[aten.uniform.default]
        # npu input -> NotImplemented
        x_npu = torch.empty(4, device="npu")
        self.assertIs(fn(x_npu, 0.0, 1.0), NotImplemented)
        # cpu input -> real tensor (original decomp runs, philox path intact)
        x_cpu = torch.empty(4, device="cpu")
        out = fn(x_cpu, 0.0, 1.0)
        self.assertIsNot(out, NotImplemented)
        self.assertIsInstance(out, torch.Tensor)

    @requires_npu
    def test_compiled_dropout_family_matches_eager_bitwise(self):
        # One representative per failing opinfo family; both sides seeded so a
        # mask-stream desync shows up as a bit mismatch, not noise.
        cases = {
            "dropout2d": lambda t: torch.nn.functional.dropout2d(t, 0.3, True),
            "dropout3d": lambda t: torch.nn.functional.dropout3d(t, 0.3, True),
            "alpha_dropout": lambda t: torch.nn.functional.alpha_dropout(t, 0.3, True),
            "feature_alpha_dropout": lambda t: torch.nn.functional.feature_alpha_dropout(
                t, 0.3, True
            ),
            "rrelu": lambda t: torch.nn.functional.rrelu(t, 0.1, 1 / 3, True),
        }
        x = torch.randn(4, 8, 16, 16, 32, device="npu")
        for name, fn in cases.items():
            with self.subTest(op=name):
                torch._dynamo.reset()
                torch.npu.manual_seed(0)
                torch.manual_seed(0)
                eager = fn(x)
                compiled = torch.compile(
                    fn,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options={"npu_backend": "triton_experimental"},
                )
                torch.npu.manual_seed(0)
                torch.manual_seed(0)
                actual, codes = run_and_get_code(compiled, x)
                self.assertEqual(actual, eager, rtol=0, atol=0)
                source = "\n".join(codes)
                self.assertNotIn(
                    "inductor_random",
                    source,
                    "philox codegen leaked back into the compiled graph",
                )


if __name__ == "__main__":
    run_tests()
