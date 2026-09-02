#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
Add validation cases for torch._C APIs on NPU:
1. PyTorch community lacks direct validation for some torch._C APIs, so this file is added.
2. This file validates:
   torch._C._nn.flatten_dense_tensors
   torch._C.DisableTorchFunction
   torch._C.DisableTorchFunctionSubclass
(extendable)

"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class _DispatchTrackingTensor(torch.Tensor):
    """Tensor subclass that bumps a counter every time __torch_function__
    is invoked, so tests can assert whether dispatch was active."""
    dispatch_count = [0]

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.dispatch_count[0] += 1
        return torch.zeros(1, device=device_type).as_subclass(cls)


class TestFlattenDenseTensors(TestCase):

    def test_flattens_2d_tensors_to_1d_concat(self):
        tensors = [torch.zeros(2, 3, device=device_type) for _ in range(3)]
        flat = torch._C._nn.flatten_dense_tensors(tensors)
        self.assertEqual(flat.shape, (18,))
        self.assertEqual(flat.device.type, device_type)

    def test_concatenates_size_1_with_size_2(self):
        tensors = [
            torch.tensor([1.0], device=device_type),
            torch.tensor([2.0, 3.0], device=device_type),
        ]
        flat = torch._C._nn.flatten_dense_tensors(tensors)
        self.assertEqual(flat.shape, (3,))

    def test_concatenated_values_match_input_order(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device_type)
        b = torch.tensor([[5.0, 6.0]], device=device_type)
        flat = torch._C._nn.flatten_dense_tensors([a, b])
        expected = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=device_type,
        )
        self.assertEqual(flat, expected)

    def test_flatten_single_tensor(self):
        tensors = [torch.tensor([1.0, 2.0, 3.0], device=device_type)]
        flat = torch._C._nn.flatten_dense_tensors(tensors)
        self.assertEqual(flat.shape, (3,))
        self.assertEqual(flat, torch.tensor([1.0, 2.0, 3.0], device=device_type))

    def test_flatten_empty_list_raises_runtime_error(self):
        with self.assertRaises((ValueError, RuntimeError)):
            torch._C._nn.flatten_dense_tensors([])

    def test_flatten_tuple_input(self):
        tensors = (
            torch.tensor([1.0], device=device_type),
            torch.tensor([2.0], device=device_type),
        )
        flat = torch._C._nn.flatten_dense_tensors(tensors)
        self.assertEqual(flat.shape, (2,))
        self.assertEqual(flat, torch.tensor([1.0, 2.0], device=device_type))

    def test_flatten_raises_for_non_tensor_element(self):
        with self.assertRaises(TypeError):
            torch._C._nn.flatten_dense_tensors([torch.tensor(1.0, device=device_type), "not_a_tensor"])

    def test_flatten_raises_for_mismatched_device(self):
        cpu_t = torch.tensor([1.0])
        npu_t = torch.tensor([2.0], device=device_type) if device_type != "cpu" else cpu_t
        if device_type != "cpu":
            with self.assertRaises(RuntimeError):
                torch._C._nn.flatten_dense_tensors([cpu_t, npu_t])


class TestDisableTorchFunction(TestCase):

    def test_disables_torch_function_on_subclass(self):
        _DispatchTrackingTensor.dispatch_count[0] = 0
        x = torch.tensor(
            [1, 2, 3], device=device_type,
        ).as_subclass(_DispatchTrackingTensor)
        # before any context: dispatch IS active
        before_count = _DispatchTrackingTensor.dispatch_count[0]
        _ = x + 1
        self.assertEqual(
            _DispatchTrackingTensor.dispatch_count[0],
            before_count + 1,
            "subclass __torch_function__ must run outside DisableTorchFunction",
        )
        # inside context: dispatch must be SUSPENDED, count does NOT advance
        ctx = torch._C.DisableTorchFunction()
        ctx.__enter__()
        try:
            inside_before = _DispatchTrackingTensor.dispatch_count[0]
            _ = x + 1
            self.assertEqual(
                _DispatchTrackingTensor.dispatch_count[0],
                inside_before,
                "__torch_function__ must NOT run inside DisableTorchFunction",
            )
        finally:
            ctx.__exit__(None, None, None)
        # after exit: dispatch must RESUME (exactly one call)
        _ = x + 1
        self.assertEqual(
            _DispatchTrackingTensor.dispatch_count[0],
            inside_before + 1,
            "subclass __torch_function__ must run again after DisableTorchFunction exit",
        )

    def test_op_inside_disable_returns_plain_tensor(self):
        """When dispatch is suspended, an op on a subclass tensor must NOT route
        through the subclass __torch_function__: the result is a plain torch.Tensor
        even though the input is a subclass."""
        _DispatchTrackingTensor.dispatch_count[0] = 0
        x = torch.tensor(
            [1, 2, 3], device=device_type,
        ).as_subclass(_DispatchTrackingTensor)
        outside_y = x + 1
        self.assertIsInstance(
            outside_y,
            _DispatchTrackingTensor,
            "outside DisableTorchFunction the subclass op routes through __torch_function__",
        )
        with torch._C.DisableTorchFunction():
            inside_y = x + 1
        self.assertNotIsInstance(
            inside_y,
            _DispatchTrackingTensor,
            "inside DisableTorchFunction dispatch is suspended so result is plain tensor",
        )
        self.assertEqual(inside_y, torch.tensor([2, 3, 4], device=device_type))


class TestDisableTorchFunctionSubclass(TestCase):

    def test_disables_torch_function_subclass_on_subclass(self):
        _DispatchTrackingTensor.dispatch_count[0] = 0
        x = torch.tensor(
            [1, 2, 3], device=device_type,
        ).as_subclass(_DispatchTrackingTensor)
        before_count = _DispatchTrackingTensor.dispatch_count[0]
        _ = x + 1
        self.assertEqual(
            _DispatchTrackingTensor.dispatch_count[0],
            before_count + 1,
            "subclass __torch_function__ must run outside DisableTorchFunctionSubclass",
        )
        ctx = torch._C.DisableTorchFunctionSubclass()
        ctx.__enter__()
        try:
            inside_before = _DispatchTrackingTensor.dispatch_count[0]
            _ = x + 1
            self.assertEqual(
                _DispatchTrackingTensor.dispatch_count[0],
                inside_before,
                "__torch_function__ must NOT run inside DisableTorchFunctionSubclass",
            )
        finally:
            ctx.__exit__(None, None, None)
        _ = x + 1
        self.assertEqual(
            _DispatchTrackingTensor.dispatch_count[0],
            inside_before + 1,
            "subclass __torch_function__ must run again after DisableTorchFunctionSubclass exit",
        )

    def test_op_inside_disable_subclass_returns_plain_tensor(self):
        """Same rationale as the non-Subclass variant: input is a subclass but
        dispatch is suspended, so the result is plain torch.Tensor (not subclass)."""
        _DispatchTrackingTensor.dispatch_count[0] = 0
        x = torch.tensor(
            [1, 2, 3], device=device_type,
        ).as_subclass(_DispatchTrackingTensor)
        outside_y = x + 1
        self.assertIsInstance(outside_y, _DispatchTrackingTensor)
        with torch._C.DisableTorchFunctionSubclass():
            inside_y = x + 1
        self.assertNotIsInstance(inside_y, _DispatchTrackingTensor)
        self.assertEqual(inside_y, torch.tensor([2, 3, 4], device=device_type))


if __name__ == "__main__":
    run_tests()
