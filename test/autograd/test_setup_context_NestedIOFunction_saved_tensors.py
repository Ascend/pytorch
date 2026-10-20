# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable Law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Add validation cases for
torch.autograd.function.InplaceFunction.setup_context and
torch.autograd.function.NestedIOFunction.saved_tensors on NPU.

This file validates setup_context with NPU tensors during autograd backward
and validates that NestedIOFunction.saved_tensors restores the original
nested structure while preserving NPU tensor values and devices.
"""

import warnings

import torch

from torch.autograd.function import Function, InplaceFunction, NestedIOFunction
from torch_npu.testing.testcase import TestCase, run_tests


class SquareWithSetupContext(InplaceFunction):
    @staticmethod
    def forward(x):
        return x * x

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return 2 * x * grad_output


class TestAutogradFunctionApi(TestCase):

    def test_inplace_function_setup_context(self):
        x_cpu = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        x_npu = x_cpu.detach().clone().npu()
        x_npu.requires_grad_(True)

        output_cpu = SquareWithSetupContext.apply(x_cpu)
        output_npu = SquareWithSetupContext.apply(x_npu)

        output_cpu.sum().backward()
        output_npu.sum().backward()

        self.assertEqual(
            output_cpu,
            output_npu.cpu(),
        )

        self.assertEqual(
            x_cpu.grad,
            x_npu.grad.cpu(),
        )

        self.assertEqual(
            output_npu.device.type,
            "npu",
        )

        self.assertEqual(
            x_npu.grad.device.type,
            "npu",
        )

    def test_nested_io_function_saved_tensors(self):
        x = torch.tensor(
            [1.0, 2.0],
            dtype=torch.float32,
            device="npu",
        )
        y = torch.tensor(
            [3.0, 4.0],
            dtype=torch.float32,
            device="npu",
        )
        z = torch.tensor(
            [5.0, 6.0],
            dtype=torch.float32,
            device="npu",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            func = NestedIOFunction()

        # save_for_backward records the original nested structure
        # and flattens all tensors into func.to_save.
        func.save_for_backward([x, (y,)], z)

        self.assertEqual(len(func.to_save), 3)

        # NestedIOFunction.saved_tensors obtains flat saved tensors
        # from Function.saved_tensors and then restores the original
        # nested structure. Since NestedIOFunction is a legacy class,
        # replace the parent descriptor temporarily to isolate and
        # validate the saved_tensors property itself.
        original_saved_tensors = Function.__dict__.get(
            "saved_tensors",
            None,
        )

        Function.saved_tensors = property(
            lambda self: self._flat_saved_tensors
        )

        try:
            func._flat_saved_tensors = func.to_save
            result = func.saved_tensors
        finally:
            if original_saved_tensors is None:
                delattr(Function, "saved_tensors")
            else:
                Function.saved_tensors = original_saved_tensors

        # Original structure:
        # ([x, (y,)], z)
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result[0], list))
        self.assertTrue(isinstance(result[0][1], tuple))

        saved_x = result[0][0]
        saved_y = result[0][1][0]
        saved_z = result[1]

        self.assertEqual(saved_x, x)
        self.assertEqual(saved_y, y)
        self.assertEqual(saved_z, z)

        self.assertEqual(saved_x.device.type, "npu")
        self.assertEqual(saved_y.device.type, "npu")
        self.assertEqual(saved_z.device.type, "npu")


if __name__ == "__main__":
    run_tests()
