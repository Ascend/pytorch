# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
# Owner(s): ["module: inductor"]
"""triton_experimental must lower aten.contiguous to a generated copy kernel.

``aten.contiguous.default`` has no upstream lowering -- unlike ``as_strided`` it
is not already in the ``lowerings`` table -- so the bulk fallback pass in
``_register_npu_inductor_fallbacks`` never sees it, and a strict-mode compile of
any graph that emits a bare contiguous node fails with
``MissingOperatorWithoutDecomp`` (issue #4933: all six diag_embed opinfo dtype
variants, whose decomposition chain ends in ``mask_tensor(...).contiguous()``).
The fix registers ``npu_contiguous`` (delegating to ``npu_clone``) and lists
``aten.contiguous`` in ``GENERATE_LIST`` so the bulk pass does not clobber that
new lowering back into a fallback.  These tests pin the registration and check
the compiled diag_embed matches eager across the six reported dtypes.
"""

import os
import unittest

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor import config
from torch._inductor.lowering import lowerings
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import (
    lowering,
    lowering_override_list,
)

aten = torch.ops.aten

requires_npu = unittest.skipUnless(torch.npu.is_available(), "requires an NPU")

CONTIGUOUS = aten.contiguous.default


class TestTeContiguousLoweringRegistration(TestCase):
    def test_contiguous_listed_for_generation(self):
        self.assertIn(aten.contiguous, lowering_override_list.GENERATE_LIST)

    def test_npu_contiguous_is_registered_as_a_lowering(self):
        self.assertIn(CONTIGUOUS, lowerings)

    def test_contiguous_is_exempted_from_the_bulk_fallback_pass(self):
        # Re-implement the exemption invariant the bulk pass uses, instead of
        # re-running the global pass (which would re-scan every lowering and
        # append to the global FALLBACK_LIST -- order-dependent and polluting).
        gen_set = set()
        for fn in lowering_override_list.GENERATE_LIST:
            gen_set.add(fn)
            if isinstance(fn, torch._ops.OpOverloadPacket):
                gen_set |= {getattr(fn, o) for o in fn.overloads()}
        self.assertIn(CONTIGUOUS, gen_set)
        self.assertNotIn(CONTIGUOUS, lowering.FALLBACK_LIST)


class TestTeDiagEmbedMatchesEager(TestCase):
    @requires_npu
    def test_diag_embed_forward(self):
        # The six dtype variants reported in issue #4933.
        for dtype in (
            torch.bool,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        ):
            with self.subTest(dtype=dtype):
                if dtype == torch.bool:
                    x = torch.randint(0, 2, (3, 4), device="npu", dtype=torch.int64).to(
                        torch.bool
                    )
                elif dtype in (torch.int32, torch.int64):
                    x = torch.randint(0, 10, (3, 4), device="npu", dtype=torch.int64).to(
                        dtype
                    )
                else:
                    x = torch.randn(3, 4, device="npu", dtype=dtype)
                eager = torch.diag_embed(x, offset=0, dim1=-2, dim2=-1)
                torch._dynamo.reset()
                try:
                    with config.patch("force_disable_caches", True), config.patch(
                        "implicit_fallbacks", False
                    ):
                        compiled = torch.compile(
                            lambda a: torch.diag_embed(a, offset=0, dim1=-2, dim2=-1),
                            backend="inductor",
                            fullgraph=True,
                            options={"npu_backend": "triton_experimental"},
                        )
                        out = compiled(x)
                finally:
                    torch._dynamo.reset()
                self.assertEqual(out.dtype, eager.dtype)
                self.assertEqual(out.shape, eager.shape)
                torch.testing.assert_close(out, eager)


if __name__ == "__main__":
    run_tests()
