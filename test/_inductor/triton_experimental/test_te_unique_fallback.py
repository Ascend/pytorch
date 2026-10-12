# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental must register a native fallback for aten._unique.

``index_fill``'s compiled backward reaches ``aten._unique.default``, which has
neither an upstream lowering nor a decomposition, so the bulk fallback pass in
``lowering.py`` (which walks ``lowerings``) cannot discover it; a TE-compiled
``index_fill`` under ``implicit_fallbacks=False`` raised
``MissingOperatorWithoutDecomp`` (target ``aten._unique.default``) while the
default backend and eager passed.  These tests pin the ``_unique``/``_unique2``
overloads in the explicit fallback list and check the compiled ``index_fill``
gradient matches eager.  Fixes #4937.
"""

import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor import config
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import lowering_fallback_list

aten = torch.ops.aten

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")


class TestTeUniqueFallback(TestCase):
    def test_unique_listed_for_explicit_fallback(self):
        for op in (
            aten._unique.default,
            aten._unique.out,
            aten._unique2.default,
            aten._unique2.out,
        ):
            self.assertIn(op, lowering_fallback_list.EXPLICIT_FALLBACK_LIST)

    def test_list_entries_are_concrete_overloads(self):
        for op in lowering_fallback_list.EXPLICIT_FALLBACK_LIST:
            self.assertIsInstance(op, torch._ops.OpOverload)

    def test_registrar_consumes_same_list_object(self):
        from torch_npu._inductor.triton_experimental import lowering

        self.assertIs(
            lowering.EXPLICIT_FALLBACK_LIST,
            lowering_fallback_list.EXPLICIT_FALLBACK_LIST,
        )

    @requires_npu
    def test_index_fill_backward_fallback(self):
        def fn(a, idx):
            return a.index_fill(0, idx, 2.0)

        idx = torch.tensor([0, 3, 3, 1], device="npu")
        eager_input = torch.randn(5, 4, device="npu").requires_grad_(True)
        fn(eager_input, idx).sum().backward()
        eager_grad = eager_input.grad.detach().clone()

        compiled = torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            options={"npu_backend": "triton_experimental"},
        )
        compiled_input = torch.randn(5, 4, device="npu").requires_grad_(True)
        try:
            with (
                config.patch("force_disable_caches", True),
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
                compiled(compiled_input, idx).sum().backward()
        finally:
            torch._dynamo.reset()

        torch.testing.assert_close(
            compiled_input.grad, eager_grad, rtol=1e-2, atol=1e-3
        )


if __name__ == "__main__":
    run_tests()
