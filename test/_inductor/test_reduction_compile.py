#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
import os
import unittest

# This case targets the Triton (default) sub-backend only: the torch_npu
# make_reduction override replaces the upstream symbol only when that backend
# is loaded.
os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "default")

import torch  # noqa: E402
from torch.testing._internal.common_utils import (  # noqa: E402
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils  # noqa: E402
import torch_npu  # noqa: E402,F401


def _loaded_backend():
    """Return the sub-backend name loaded by torch_npu._inductor, or None."""
    try:
        from torch_npu.utils._dynamo import _InductorNpuRegistry

        return getattr(_InductorNpuRegistry, "_loaded_backend", None)
    except (ImportError, AttributeError):
        # The registry moved or is unavailable: let the case run, a real
        # backend mismatch still surfaces as a compile failure.
        return None


_BACKEND = _loaded_backend()


@unittest.skipUnless(
    _BACKEND in (None, "default"),
    "make_reduction override only applies to the default (Triton) sub-backend, "
    "but %s is loaded" % _BACKEND,
)
class TestReductionCompile(TestUtils):
    """Regression tests for compiling dim reductions on the default Inductor backend.

    torch_npu replaces torch._inductor.lowering.make_reduction with its own
    implementation.  That override kept the older signature and therefore did
    not accept the strict_reduction keyword, while the upstream sum_ lowering
    passes it unconditionally.  Compiling any aten.sum(dim=...) graph then
    raised TypeError during lowering.

    Only the dim variant is covered here: the full-reduction forms
    (``torch.sum(t)`` / ``torch.mean(t)``) additionally hit an unrelated
    scheduler defect, tracked separately.
    """

    def op_calc(self, tensor, dim):
        return torch.sum(tensor, dim)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dim', [0, -1])
    @parametrize('dtype', ['float32'])
    def test_sum_with_dim(self, shape, dim, dtype):
        element = self._generate_tensor(shape, dtype)
        std_sum = self.op_calc(element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_sum = compiled_op_calc(element, dim)

        # Same tolerance as the other inductor cases in this directory.
        self.assertEqual(std_sum, inductor_sum, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestReductionCompile)

if __name__ == "__main__":
    run_tests()
