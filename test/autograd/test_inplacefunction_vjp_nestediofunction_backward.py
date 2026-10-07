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
# See the License for the specific Language governing permissions and
# limitations under the License.
"""
Add validation cases for
torch.autograd.function.InplaceFunction.vjp and
torch.autograd.function.NestedIOFunction.backward on NPU.

This file validates InplaceFunction.vjp with NPU tensors during autograd
backward, and validates NestedIOFunction.backward rebuilds nested gradient
structures and flattens returned gradients while preserving NPU tensor
values and devices.
"""

import warnings

import torch
import torch_npu
from torch.autograd.function import InplaceFunction, NestedIOFunction
from torch_npu.testing.testcase import TestCase, run_tests


class InplaceScaleVJP(InplaceFunction):
    """Custom InplaceFunction that defines vjp instead of backward."""

    vjp_called = False

    @staticmethod
    def forward(ctx, input_tensor, scale):
        ctx.scale = scale
        ctx.mark_dirty(input_tensor)
        input_tensor.mul_(scale)
        return input_tensor

    @staticmethod
    def vjp(ctx, grad_output):
        InplaceScaleVJP.vjp_called = True
        grad_input = grad_output * ctx.scale
        return grad_input, None


class NestedBackwardFunction(NestedIOFunction):
    """Helper subclass used to validate NestedIOFunction.backward."""

    def __init__(self):
        super().__init__()
        self.received_gradients = None

    def backward_extended(self, grad_first, grad_nested):
        self.received_gradients = (grad_first, grad_nested)

        grad_input_0 = grad_first * 2
        grad_input_1 = grad_nested[0] * 3
        grad_input_2 = None
        grad_input_3 = grad_nested[1][1] * 4

        return grad_input_0, [grad_input_1, grad_input_2, grad_input_3]


class TestAutogradFunctionVjpBackward(TestCase):
    def test_inplace_function_vjp(self):
        # Use a non-leaf tensor because the custom Function performs an
        # in-place operation and marks the input dirty.
        base = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float32,
            device="npu",
            requires_grad=True,
        )
        input_tensor = base.clone()
        grad_output = torch.tensor(
            [0.5, 1.5, 2.0],
            dtype=torch.float32,
            device="npu",
        )

        InplaceScaleVJP.vjp_called = False

        output = InplaceScaleVJP.apply(input_tensor, 3.0)
        output.backward(grad_output)

        expected_output = torch.tensor(
            [3.0, 6.0, 9.0],
            dtype=torch.float32,
        )
        expected_grad = torch.tensor(
            [1.5, 4.5, 6.0],
            dtype=torch.float32,
        )

        self.assertTrue(InplaceScaleVJP.vjp_called)

        self.assertEqual(output.device.type, "npu")
        self.assertEqual(base.grad.device.type, "npu")

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(base.grad.dtype, torch.float32)

        self.assertRtolEqual(output.cpu(), expected_output)
        self.assertRtolEqual(base.grad.cpu(), expected_grad)

    def test_nested_io_function_backward(self):
        # NestedIOFunction is a backward-compatibility utility. Its backward()
        # uses _nested_output as a structure prototype to rebuild flat incoming
        # gradients before calling backward_extended().
        prototype_0 = torch.empty(
            2,
            dtype=torch.float32,
            device="npu",
        )
        prototype_1 = torch.empty(
            2,
            dtype=torch.float32,
            device="npu",
        )
        prototype_2 = torch.empty(
            2,
            dtype=torch.float32,
            device="npu",
        )

        grad_0 = torch.tensor(
            [1.0, 2.0],
            dtype=torch.float32,
            device="npu",
        )
        grad_1 = torch.tensor(
            [3.0, 4.0],
            dtype=torch.float32,
            device="npu",
        )
        grad_2 = torch.tensor(
            [5.0, 6.0],
            dtype=torch.float32,
            device="npu",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            function = NestedBackwardFunction()

        function._nested_output = (
            prototype_0,
            [
                prototype_1,
                (None, prototype_2),
            ],
        )

        result = function.backward(
            grad_0,
            grad_1,
            grad_2,
        )

        received_first, received_nested = function.received_gradients

        self.assertTrue(isinstance(received_nested, list))
        self.assertTrue(isinstance(received_nested[1], tuple))
        self.assertIsNone(received_nested[1][0])

        self.assertRtolEqual(
            received_first.cpu(),
            grad_0.cpu(),
        )
        self.assertRtolEqual(
            received_nested[0].cpu(),
            grad_1.cpu(),
        )
        self.assertRtolEqual(
            received_nested[1][1].cpu(),
            grad_2.cpu(),
        )

        self.assertEqual(len(result), 4)
        self.assertIsNone(result[2])

        expected_grad_0 = torch.tensor(
            [2.0, 4.0],
            dtype=torch.float32,
        )
        expected_grad_1 = torch.tensor(
            [9.0, 12.0],
            dtype=torch.float32,
        )
        expected_grad_3 = torch.tensor(
            [20.0, 24.0],
            dtype=torch.float32,
        )

        self.assertRtolEqual(
            result[0].cpu(),
            expected_grad_0,
        )
        self.assertRtolEqual(
            result[1].cpu(),
            expected_grad_1,
        )
        self.assertRtolEqual(
            result[3].cpu(),
            expected_grad_3,
        )

        self.assertEqual(result[0].device.type, "npu")
        self.assertEqual(result[1].device.type, "npu")
        self.assertEqual(result[3].device.type, "npu")

        self.assertEqual(result[0].dtype, torch.float32)
        self.assertEqual(result[1].dtype, torch.float32)
        self.assertEqual(result[3].dtype, torch.float32)


if __name__ == "__main__":
    print('torch_npu version is ',torch_npu.__version__)
    print('torch version is', torch.__version__)
    current_device = torch.npu.current_device()
    device_name = torch.npu.get_device_name(current_device)
    print(f"Current NPU name   : {device_name}")
    run_tests()
