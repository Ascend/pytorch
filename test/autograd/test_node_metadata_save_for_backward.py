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
Add validation cases for torch.autograd.graph.Node.metadata and
torch.autograd.FunctionCtx.save_for_backward on NPU.

This file validates that Autograd graph node metadata can be accessed and
mutated on NPU-created grad_fn nodes, and that custom autograd Functions can
save NPU tensors for backward through ctx.save_for_backward.
"""

import torch
from torch_npu.testing.testcase import TestCase, run_tests


DEVICE = "npu"


def _npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


class TestAutogradGraphNodeMetadata(TestCase):

    def setUp(self):
        super().setUp()
        if not _npu_available():
            self.skipTest("NPU is not available")

    def test_node_metadata_returns_dict_on_npu(self):
        x = torch.arange(1, 7, dtype=torch.float32, device=DEVICE)
        x = x.reshape(2, 3).clone().detach().requires_grad_(True)

        y = (x * 2).sum()
        metadata = y.grad_fn.metadata

        self.assertIsInstance(metadata, dict)

    def test_node_metadata_can_be_mutated_on_npu(self):
        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)

        y = (x + 1).sum()
        node = y.grad_fn

        node.metadata["case"] = "npu_metadata"
        node.metadata["numel"] = x.numel()

        self.assertEqual(node.metadata["case"], "npu_metadata")
        self.assertEqual(node.metadata["numel"], 4)

    def test_node_metadata_does_not_affect_backward_on_npu(self):
        x = torch.arange(1, 7, dtype=torch.float32, device=DEVICE)
        x = x.reshape(2, 3).clone().detach().requires_grad_(True)

        y = (x * x).sum()
        node = y.grad_fn
        node.metadata["tag"] = "square_sum"

        y.backward()

        expected_grad = 2 * x.detach()
        self.assertEqual(x.grad.cpu(), expected_grad.cpu())
        self.assertEqual(node.metadata["tag"], "square_sum")

    def test_node_metadata_access_after_backward_on_npu(self):
        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)

        y = (x * 3).sum()
        node = y.grad_fn
        node.metadata["stage"] = "before_backward"

        y.backward()

        self.assertIsInstance(node.metadata, dict)
        self.assertEqual(node.metadata["stage"], "before_backward")

    def test_node_metadata_multiple_nodes_are_independent_on_npu(self):
        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)

        y1 = x + 1
        y2 = x * 2

        node1 = y1.grad_fn
        node2 = y2.grad_fn

        node1.metadata["node"] = "add"
        node2.metadata["node"] = "mul"

        self.assertEqual(node1.metadata["node"], "add")
        self.assertEqual(node2.metadata["node"], "mul")


class TestAutogradFunctionCtxSaveForBackward(TestCase):

    def setUp(self):
        super().setUp()
        if not _npu_available():
            self.skipTest("NPU is not available")

    def test_save_for_backward_single_tensor_on_npu(self):
        class SquareFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return 2 * x * grad_output

        x = torch.arange(1, 7, dtype=torch.float32, device=DEVICE)
        x = x.reshape(2, 3).clone().detach().requires_grad_(True)

        y = SquareFunction.apply(x).sum()
        y.backward()

        expected_grad = 2 * x.detach()
        self.assertEqual(x.grad.cpu(), expected_grad.cpu())

    def test_save_for_backward_multiple_tensors_on_npu(self):
        class MulAddFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x * y + y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                grad_x = grad_output * y
                grad_y = grad_output * (x + 1)
                return grad_x, grad_y

        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        y = torch.arange(5, 9, dtype=torch.float32, device=DEVICE)

        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        out = MulAddFunction.apply(x, y).sum()
        out.backward()

        expected_grad_x = y.detach()
        expected_grad_y = x.detach() + 1

        self.assertEqual(x.grad.cpu(), expected_grad_x.cpu())
        self.assertEqual(y.grad.cpu(), expected_grad_y.cpu())

    def test_save_for_backward_with_none_on_npu(self):
        class SaveNoneFunction(torch.autograd.Function):
            seen_none = False

            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(None, x, None)
                return x * x

            @staticmethod
            def backward(ctx, grad_output):
                n1, x, n2 = ctx.saved_tensors
                SaveNoneFunction.seen_none = n1 is None and n2 is None
                return 2 * x * grad_output

        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)

        out = SaveNoneFunction.apply(x).sum()
        out.backward()

        expected_grad = 2 * x.detach()
        self.assertTrue(SaveNoneFunction.seen_none)
        self.assertEqual(x.grad.cpu(), expected_grad.cpu())

    def test_save_for_backward_with_non_tensor_ctx_attribute_on_npu(self):
        class ScaleFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, scale):
                ctx.save_for_backward(x)
                ctx.scale = scale
                return x * scale

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * ctx.scale, None

        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)

        scale = 3.0
        out = ScaleFunction.apply(x, scale).sum()
        out.backward()

        expected_grad = torch.full_like(x, scale)
        self.assertEqual(x.grad.cpu(), expected_grad.cpu())

    def test_save_for_backward_triggers_saved_tensor_hooks_on_npu(self):
        class HookFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return 2 * x * grad_output

        pack_devices = []
        unpack_devices = []

        def pack_hook(tensor):
            pack_devices.append(tensor.device.type)
            return tensor.detach().clone()

        def unpack_hook(tensor):
            unpack_devices.append(tensor.device.type)
            return tensor

        x = torch.arange(1, 5, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out = HookFunction.apply(x).sum()
            out.backward()

        expected_grad = 2 * x.detach()

        self.assertEqual(x.grad.cpu(), expected_grad.cpu())
        self.assertEqual(pack_devices, ["npu"])
        self.assertEqual(unpack_devices, ["npu"])

    def test_save_for_backward_detects_inplace_modified_tensor_on_npu(self):
        class SaveInputFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * x

        x = torch.ones(4, dtype=torch.float32, device=DEVICE, requires_grad=True)
        out = SaveInputFunction.apply(x)

        with torch.no_grad():
            x.add_(1)

        with self.assertRaisesRegex(
            RuntimeError,
            "modified by an inplace operation|one of the variables needed",
        ):
            out.sum().backward()


if __name__ == "__main__":
    run_tests()
