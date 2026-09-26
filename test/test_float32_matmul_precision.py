# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch global matmul precision config APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.set_float32_matmul_precision, torch.get_float32_matmul_precision (extendable).
"""

import os
import warnings

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


_VALID_PRECISIONS = ("highest", "high", "medium")


class TestFloat32MatmulPrecision(TestCase):
    def test_float32_matmul_precision_default_highest(self):
        # Default is "highest" unless TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set,
        # which makes the global default "high" (see aten/src/ATen/Context.h).
        skip_tf32_cublas = (
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in os.environ
            and int(os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"])
        )
        if not skip_tf32_cublas:
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_float32_matmul_precision_get_set_roundtrip(self):
        # Core get/set roundtrip for every documented precision value.
        orig = torch.get_float32_matmul_precision()
        for precision in _VALID_PRECISIONS:
            torch.set_float32_matmul_precision(precision)
            self.assertEqual(torch.get_float32_matmul_precision(), precision)
        torch.set_float32_matmul_precision(orig)
        self.assertEqual(torch.get_float32_matmul_precision(), orig)

    def test_float32_matmul_precision_invalid_value_warns(self):
        # Invalid values only warn and keep the previous precision (upstream C++
        # setFloat32MatmulPrecision does not raise, see aten/src/ATen/Context.cpp).
        orig = torch.get_float32_matmul_precision()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.set_float32_matmul_precision("invalid")
        self.assertEqual(torch.get_float32_matmul_precision(), orig)
        self.assertTrue(any("is not one of" in str(w.message) for w in caught))

    def test_float32_matmul_precision_npu_hf32_switch_is_allow_hf32(self):
        # NPU exposes the independent HF32 matmul switch as allow_hf32 (via
        # torch_npu.npu.matmul, an _allowHF32Matmul instance), not the CUDA-only
        # allow_tf32. The legacy NPU port referenced allow_tf32, which resolves
        # to None through _allowHF32Matmul.__getattr__, so its
        # assertTrue(allow_tf32) after setting "high"/"medium" failed.
        self.assertIsInstance(torch_npu.npu.matmul.allow_hf32, bool)
        self.assertIsNone(torch_npu.npu.matmul.allow_tf32)

    def test_float32_matmul_precision_independent_of_npu_allow_hf32(self):
        # torch.set_float32_matmul_precision only affects CUDA/MKLDNN precision
        # per upstream docs ("This flag currently only affects one native device
        # type: CUDA"). NPU matmul precision is controlled by the independent
        # torch_npu.npu.matmul.allow_hf32 switch, so flipping the global
        # precision flag must not touch allow_hf32.
        hf32_orig = torch_npu.npu.matmul.allow_hf32
        precision_orig = torch.get_float32_matmul_precision()
        for precision in _VALID_PRECISIONS:
            torch.set_float32_matmul_precision(precision)
            self.assertEqual(torch_npu.npu.matmul.allow_hf32, hf32_orig)
        current = torch.get_float32_matmul_precision()
        torch_npu.npu.matmul.allow_hf32 = not hf32_orig
        try:
            self.assertEqual(torch.get_float32_matmul_precision(), current)
        finally:
            torch_npu.npu.matmul.allow_hf32 = hf32_orig
            torch.set_float32_matmul_precision(precision_orig)


if __name__ == "__main__":
    run_tests()
