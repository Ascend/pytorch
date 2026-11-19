# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import os
import unittest
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "default")

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.codegen.scheduling import NPUNoLinearTritonScheduling
from torch_npu.utils._dynamo import has_triton


@unittest.skipUnless(torch.npu.is_available() and has_triton(), "requires NPU Triton")
@unittest.skipUnless(
    os.environ["TORCHINDUCTOR_NPU_BACKEND"] == "default",
    "requires the default NPU Inductor backend",
)
class TestReductionCodegenRegressions(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(config.patch({
            "combo_kernels": False,
            "compile_threads": 1,
            "triton.persistent_reductions": True,
            "force_disable_caches": True,
        }))
        # Correct eager results after fallback would hide invalid native index DSL.
        self.enterContext(mock.patch.object(
            NPUNoLinearTritonScheduling,
            "codegen_node",
            side_effect=AssertionError("Native index codegen unexpectedly fell back"),
        ))

    def test_persistent_multi_axis_reduction_store(self):
        def fn(value):
            return value.sum(dim=(0, 2))

        torch.manual_seed(0)
        # Keep 8 * 128 reduction elements within the A2/A3 persistent threshold.
        value = torch.randn(8, 64, 128, device="npu")
        expected = fn(value)
        actual = torch.compile(fn, fullgraph=True)(value)
        self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)

    @config.patch("test_configs.runtime_triton_dtype_assert", True)
    def test_arg_reduction_index_dtype(self):
        def fn(value):
            return value.argmax(dim=-1), value.argmin(dim=-1)

        torch.manual_seed(0)
        value = torch.randn(32, 256, device="npu")
        expected = fn(value)
        actual = torch.compile(fn, fullgraph=True)(value)
        self.assertEqual(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    run_tests()
