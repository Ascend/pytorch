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
Add validation cases for torch._higher_order_ops APIs on NPU:
1. PyTorch community lacks direct accelerator validation for
   torch._higher_order_ops.auto_functionalized, so this file is added.
2. This file validates the exported object and functional cloning behavior
   for mutable Tensor, optional Tensor, and Tensor list arguments (extendable).
"""

import torch
import torch._higher_order_ops
import torch._higher_order_ops.auto_functionalize
from torch.testing._internal.common_utils import run_tests, TestCase


DEVICE_TYPE = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestAutoFunctionalized(TestCase):
    def test_reexported_object_is_identical(self):
        self.assertIs(
            torch._higher_order_ops.auto_functionalized,
            torch._higher_order_ops.auto_functionalize.auto_functionalized,
        )

    def test_mutable_tensor_is_cloned_and_returned_on_device(self):
        with torch.library._scoped_library(
            "test_auto_functionalized", "FRAGMENT"
        ) as lib:
            lib.define(
                "mutate_and_sum(Tensor(a!) mutable, Tensor increment) -> Tensor"
            )

            def mutate_and_sum(mutable, increment):
                mutable.add_(increment)
                return mutable.sum()

            lib.impl(
                "mutate_and_sum", mutate_and_sum, "CompositeExplicitAutograd"
            )

            mutable = torch.tensor([1.0, 2.0, 3.0], device=DEVICE_TYPE)
            increment = torch.tensor([4.0, 5.0, 6.0], device=DEVICE_TYPE)
            original = mutable.clone()

            result, updated_mutable = torch._higher_order_ops.auto_functionalized(
                torch.ops.test_auto_functionalized.mutate_and_sum.default,
                mutable=mutable,
                increment=increment,
            )

            expected = original + increment
            self.assertEqual(mutable, original)
            self.assertEqual(updated_mutable, expected)
            self.assertEqual(result, expected.sum())
            self.assertEqual(result.device.type, DEVICE_TYPE)
            self.assertEqual(updated_mutable.device.type, DEVICE_TYPE)
            self.assertNotEqual(updated_mutable.data_ptr(), mutable.data_ptr())

    def test_mutable_tensor_list_is_cloned_on_device(self):
        with torch.library._scoped_library(
            "test_auto_functionalized", "FRAGMENT"
        ) as lib:
            lib.define(
                "mutate_list(Tensor(a!)[] mutable, Tensor increment) -> ()"
            )

            def mutate_list(mutable, increment):
                for tensor in mutable:
                    tensor.add_(increment)

            lib.impl("mutate_list", mutate_list, "CompositeExplicitAutograd")

            increment = torch.tensor([5.0, 6.0], device=DEVICE_TYPE)
            cases = [
                [],
                [
                    torch.tensor([1.0, 2.0], device=DEVICE_TYPE),
                    torch.tensor([3.0, 4.0], device=DEVICE_TYPE),
                ],
            ]
            for mutable in cases:
                with self.subTest(length=len(mutable)):
                    original = [tensor.clone() for tensor in mutable]

                    result, updated_mutable = (
                        torch._higher_order_ops.auto_functionalized(
                            torch.ops.test_auto_functionalized.mutate_list.default,
                            mutable=mutable,
                            increment=increment,
                        )
                    )

                    self.assertIsNone(result)
                    self.assertEqual(mutable, original)
                    self.assertEqual(
                        updated_mutable,
                        [tensor + increment for tensor in original],
                    )
                    for updated, source in zip(updated_mutable, mutable):
                        self.assertEqual(updated.device.type, DEVICE_TYPE)
                        self.assertNotEqual(
                            updated.data_ptr(), source.data_ptr()
                        )

    def test_multiple_outputs_precede_mutable_tensors(self):
        with torch.library._scoped_library(
            "test_auto_functionalized", "FRAGMENT"
        ) as lib:
            lib.define(
                "mutate_pair(Tensor(a!) first, Tensor(b!) second, "
                "Tensor increment) -> (Tensor, Tensor)"
            )

            def mutate_pair(first, second, increment):
                first.add_(increment)
                second.mul_(increment)
                return first.sum(), second.sum()

            lib.impl("mutate_pair", mutate_pair, "CompositeExplicitAutograd")

            first = torch.tensor([1.0, 2.0], device=DEVICE_TYPE)
            second = torch.tensor([3.0, 4.0], device=DEVICE_TYPE)
            increment = torch.tensor([2.0, 3.0], device=DEVICE_TYPE)
            original_first = first.clone()
            original_second = second.clone()

            first_result, second_result, updated_first, updated_second = (
                torch._higher_order_ops.auto_functionalized(
                    torch.ops.test_auto_functionalized.mutate_pair.default,
                    first=first,
                    second=second,
                    increment=increment,
                )
            )

            expected_first = original_first + increment
            expected_second = original_second * increment
            self.assertEqual(first, original_first)
            self.assertEqual(second, original_second)
            self.assertEqual(first_result, expected_first.sum())
            self.assertEqual(second_result, expected_second.sum())
            self.assertEqual(updated_first, expected_first)
            self.assertEqual(updated_second, expected_second)
            self.assertEqual(updated_first.device.type, DEVICE_TYPE)
            self.assertEqual(updated_second.device.type, DEVICE_TYPE)

    def test_optional_mutable_tensor_supports_none_and_tensor(self):
        with torch.library._scoped_library(
            "test_auto_functionalized", "FRAGMENT"
        ) as lib:
            lib.define(
                "mutate_optional(Tensor(a!)? mutable, Tensor increment) -> ()"
            )

            def mutate_optional(mutable, increment):
                if mutable is not None:
                    mutable.add_(increment)

            lib.impl(
                "mutate_optional", mutate_optional, "CompositeExplicitAutograd"
            )

            increment = torch.tensor([2.0, 3.0], device=DEVICE_TYPE)
            for mutable in (
                None,
                torch.tensor([1.0, 1.0], device=DEVICE_TYPE),
            ):
                with self.subTest(is_none=mutable is None):
                    original = mutable.clone() if mutable is not None else None
                    result, updated_mutable = (
                        torch._higher_order_ops.auto_functionalized(
                            torch.ops.test_auto_functionalized.mutate_optional.default,
                            mutable=mutable,
                            increment=increment,
                        )
                    )

                    self.assertIsNone(result)
                    if mutable is None:
                        self.assertIsNone(updated_mutable)
                    else:
                        self.assertEqual(mutable, original)
                        self.assertEqual(updated_mutable, original + increment)
                        self.assertEqual(updated_mutable.device.type, DEVICE_TYPE)

    def test_non_mutating_operator_is_rejected(self):
        with torch.library._scoped_library(
            "test_auto_functionalized", "FRAGMENT"
        ) as lib:
            lib.define("identity(Tensor value) -> Tensor")
            value = torch.tensor([1.0], device=DEVICE_TYPE)

            with self.assertRaises(AssertionError):
                torch._higher_order_ops.auto_functionalized(
                    torch.ops.test_auto_functionalized.identity.default,
                    value=value,
                )


if __name__ == "__main__":
    run_tests()
