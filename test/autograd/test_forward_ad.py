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
Add validation cases for torch.autograd.forward_ad APIs on NPU:
1. PyTorch community does not explicitly validate make_dual's level argument,
   and its direct cases create CPU tensors.
2. This file validates torch.autograd.forward_ad.make_dual (extendable).
"""

import importlib.util

import torch
import torch.autograd.forward_ad as fwAD
from torch.testing._internal.common_utils import run_tests, TestCase

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401


device_type = "npu"


class TestForwardAD(TestCase):

    def setUp(self):
        super().setUp()
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("Requires an available Ascend NPU")

    def test_make_dual_level_and_jvp(self):
        primal = torch.tensor([1.0, 2.0, 3.0]).to(device_type)
        tangent = torch.tensor([0.5, -1.0, 2.0]).to(device_type)
        self.assertEqual(primal.device.type, "npu")
        self.assertEqual(tangent.device.type, "npu")

        with fwAD.dual_level() as level:
            for level_arg in ("omitted", "none", "explicit"):
                with self.subTest(level_arg=level_arg):
                    if level_arg == "omitted":
                        dual = fwAD.make_dual(primal, tangent)
                    elif level_arg == "none":
                        dual = fwAD.make_dual(primal, tangent, level=None)
                    else:
                        dual = fwAD.make_dual(primal, tangent, level=level)

                    unpacked = fwAD.unpack_dual(dual, level=level)
                    self.assertEqual(unpacked.primal, primal)
                    self.assertEqual(unpacked.tangent, tangent)

                    output_primal, output_tangent = fwAD.unpack_dual(
                        dual * dual, level=level
                    )
                    self.assertEqual(output_primal, primal * primal)
                    self.assertEqual(output_tangent, 2 * primal * tangent)

    def test_make_dual_noncontiguous_tangent(self):
        primal = torch.arange(6, dtype=torch.float32).to(device_type).reshape(2, 3)
        tangent = (
            torch.arange(6, dtype=torch.float32)
            .to(device_type)
            .reshape(3, 2)
            .t()
        )
        self.assertFalse(tangent.is_contiguous())

        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)
            unpacked = fwAD.unpack_dual(dual)
            self.assertEqual(unpacked.primal, primal)
            self.assertEqual(unpacked.tangent, tangent)
            output = fwAD.unpack_dual(dual * dual)
            self.assertEqual(output.primal, primal * primal)
            self.assertEqual(output.tangent, 2 * primal * tangent)

    def test_make_dual_backward_differentiable(self):
        primal = torch.tensor([1.0, 2.0], requires_grad=True).to(device_type)
        tangent = torch.tensor([3.0, 4.0], requires_grad=True).to(device_type)

        with fwAD.dual_level():
            unpacked = fwAD.unpack_dual(fwAD.make_dual(primal, tangent))
            primal_grad, tangent_grad = torch.autograd.grad(
                unpacked.primal.sum(),
                (primal, tangent),
                allow_unused=True,
            )
            self.assertEqual(primal_grad, torch.ones_like(primal))
            self.assertIsNone(tangent_grad)

            primal_grad, tangent_grad = torch.autograd.grad(
                unpacked.tangent.sum(),
                (primal, tangent),
                allow_unused=True,
            )
            self.assertIsNone(primal_grad)
            self.assertEqual(tangent_grad, torch.ones_like(tangent))

    def test_make_dual_empty_tensor(self):
        primal = torch.empty(0).to(device_type)
        tangent = torch.empty(0).to(device_type)

        with fwAD.dual_level():
            unpacked = fwAD.unpack_dual(fwAD.make_dual(primal, tangent))
            self.assertEqual(unpacked.primal, primal)
            self.assertEqual(unpacked.tangent, tangent)

    def test_make_dual_no_active_level_error(self):
        primal = torch.ones(2).to(device_type)
        tangent = torch.ones(2).to(device_type)

        with self.assertRaisesRegex(RuntimeError, "no level exists"):
            fwAD.make_dual(primal, tangent)

    def test_make_dual_shape_mismatch_error(self):
        primal = torch.ones(2).to(device_type)
        tangent = torch.ones(3).to(device_type)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "different size"):
                fwAD.make_dual(primal, tangent)

    def test_make_dual_integral_dtype_errors(self):
        float_tensor = torch.ones(2).to(device_type)
        int_tensor = torch.ones(2, dtype=torch.int64).to(device_type)

        with fwAD.dual_level():
            with self.assertRaisesRegex(ValueError, "Expected primal"):
                fwAD.make_dual(int_tensor, float_tensor)
            with self.assertRaisesRegex(ValueError, "Expected tangent"):
                fwAD.make_dual(float_tensor, int_tensor)


if __name__ == "__main__":
    run_tests()
