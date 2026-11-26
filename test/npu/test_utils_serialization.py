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
Add validation cases for torch serialization APIs on NPU:
1. Community serialization tests do not fully cover direct NPU storage reconstruction.
2. This file validates torch._utils._rebuild_tensor_v2 (extendable).
"""

import unittest
from collections import OrderedDict
from itertools import product

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


_acc = torch.accelerator.current_accelerator()
HAS_NPU = _acc is not None and _acc.type == "npu"
device_type = _acc.type if HAS_NPU else "cpu"


@unittest.skipUnless(HAS_NPU, "NPU not available")
class TestSerialization(TestCase):
    def assert_rebuilt(self, rebuilt, expected, requires_grad, hooks):
        self.assertEqual(rebuilt.device.type, "npu")
        self.assertEqual(rebuilt.device, expected.device)
        self.assertEqual(rebuilt.dtype, expected.dtype)
        self.assertEqual(rebuilt.size(), expected.size())
        self.assertEqual(rebuilt.stride(), expected.stride())
        self.assertEqual(rebuilt.storage_offset(), expected.storage_offset())
        self.assertEqual(rebuilt.requires_grad, requires_grad)
        self.assertIs(rebuilt._backward_hooks, hooks)
        self.assertEqual(rebuilt.is_neg(), expected.is_neg())
        self.assertEqual(rebuilt.is_conj(), expected.is_conj())
        self.assertEqual(
            rebuilt.untyped_storage().data_ptr(),
            expected.untyped_storage().data_ptr(),
        )
        self.assertEqual(rebuilt.cpu(), expected.cpu())

    def test_rebuild_tensor_v2_layouts(self):
        base = torch.arange(40, dtype=torch.float32, device=device_type).reshape(5, 8)
        tensors = (base, base[1:5:2, 1:8:2], base[:0], base[0, 0], base[:1].expand(3, 8))
        for tensor, sequence in product(tensors, (tuple, list, torch.Size)):
            with self.subTest(shape=tensor.shape, sequence=sequence):
                rebuilt = torch._utils._rebuild_tensor_v2(
                    tensor._typed_storage(), tensor.storage_offset(),
                    sequence(tensor.size()), sequence(tensor.stride()), False, None,
                )
                self.assert_rebuilt(rebuilt, tensor, False, None)
        empty = torch.empty(0, device=device_type)
        rebuilt = torch._utils._rebuild_tensor_v2(empty._typed_storage(), 0, (0,), (1,), False, None)
        self.assert_rebuilt(rebuilt, empty, False, None)

    def test_rebuild_tensor_v2_options(self):
        tensor = torch.arange(4, dtype=torch.float32, device=device_type)
        hooks_cases = (None, OrderedDict(), OrderedDict([(0, lambda grad: grad * 2)]))
        for requires_grad, hooks, metadata in product(
            (False, True), hooks_cases, ((), (None,), ({},)),
        ):
            with self.subTest(requires_grad=requires_grad, hooks=hooks, metadata=metadata):
                rebuilt = torch._utils._rebuild_tensor_v2(
                    tensor._typed_storage(), 0, (4,), (1,), requires_grad, hooks, *metadata,
                )
                self.assert_rebuilt(rebuilt, tensor, requires_grad, hooks)
                if requires_grad:
                    rebuilt.sum().backward()
                    self.assertEqual(rebuilt.grad, torch.full_like(tensor, 2 if hooks else 1))

    def test_rebuild_tensor_v2_metadata(self):
        tensor = torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.complex64, device=device_type)
        for negative, conjugate in product((False, True), repeat=2):
            with self.subTest(negative=negative, conjugate=conjugate):
                expected = tensor[1:]
                if negative:
                    expected = torch._neg_view(expected)
                if conjugate:
                    expected = expected.conj()
                metadata = torch._utils.get_tensor_metadata(expected)
                rebuilt = torch._utils._rebuild_tensor_v2(
                    tensor._typed_storage(), 1, (2,), (1,), False, None, metadata=metadata,
                )
                self.assert_rebuilt(rebuilt, expected, False, None)

    def test_rebuild_tensor_v2_default_stride(self):
        tensor = torch.zeros(4, device=device_type)
        for stride in (None, (), []):
            with self.subTest(stride=stride):
                rebuilt = torch._utils._rebuild_tensor_v2(
                    tensor._typed_storage(), 0, (4,), stride, False, None,
                )
                self.assert_rebuilt(rebuilt, tensor, False, None)

    def test_rebuild_tensor_v2_storage_alias(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                tensor = torch.zeros(4, dtype=dtype, device=device_type)
                rebuilt = torch._utils._rebuild_tensor_v2(
                    storage=tensor._typed_storage(), storage_offset=1,
                    size=(2,), stride=(1,), requires_grad=False, backward_hooks=None,
                )
                self.assert_rebuilt(rebuilt, tensor[1:3], False, None)
                rebuilt.fill_(1)
                self.assertEqual(tensor[1:3], torch.ones(2, dtype=dtype, device=device_type))
                tensor[1:3].zero_()
                self.assertEqual(rebuilt, torch.zeros(2, dtype=dtype, device=device_type))

    def test_rebuild_tensor_v2_invalid_arguments(self):
        tensor = torch.zeros(4, device=device_type)
        arguments = [tensor._typed_storage(), 0, (4,), (1,), False, None, None]
        cases = (
            (0, None, AttributeError),
            (0, tensor.untyped_storage(), AttributeError),
            (1, -1, RuntimeError),
            (1, None, TypeError),
            (2, (-1,), RuntimeError),
            (2, None, TypeError),
            (2, 4, TypeError),
            (3, (1, 1), RuntimeError),
            (3, 1, TypeError),
            (4, None, RuntimeError),
            (4, 1, RuntimeError),
            (6, [True], AssertionError),
            (6, {"neg": "invalid"}, TypeError),
        )
        for index, value, error in cases:
            with self.subTest(argument=index, value=value):
                invalid = arguments.copy()
                invalid[index] = value
                with self.assertRaises(error):
                    torch._utils._rebuild_tensor_v2(*invalid)
        integer = torch.zeros(4, dtype=torch.int64, device=device_type)
        with self.assertRaisesRegex(RuntimeError, "require gradients"):
            torch._utils._rebuild_tensor_v2(integer._typed_storage(), 0, (4,), (1,), True, None)


if __name__ == "__main__":
    run_tests()
