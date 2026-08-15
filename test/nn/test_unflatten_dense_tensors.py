# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.nn APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch._C._nn.unflatten_dense_tensors (extendable).
"""

import torch
from torch_npu.testing.testcase import TestCase, run_tests

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestUnflattenDenseTensors(TestCase):
    def test_unflatten_dense_tensors_basic(self):
        flat = torch.arange(
            12,
            dtype=torch.float32,
            device=device_type,
        )

        tensors = [
            torch.empty(
                (2, 3),
                dtype=torch.float32,
                device=device_type,
            ),
            torch.empty(
                (3, 2),
                dtype=torch.float32,
                device=device_type,
            ),
        ]

        outputs = torch._C._nn.unflatten_dense_tensors(
            flat,
            tensors,
        )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)

        self.assertEqual(tuple(outputs[0].shape), (2, 3))
        self.assertEqual(tuple(outputs[1].shape), (3, 2))

        self.assertEqual(outputs[0].device.type, device_type)
        self.assertEqual(outputs[1].device.type, device_type)

        self.assertEqual(outputs[0].dtype, torch.float32)
        self.assertEqual(outputs[1].dtype, torch.float32)

        expected_first = torch.arange(
            0,
            6,
            dtype=torch.float32,
            device=device_type,
        ).reshape(2, 3)

        expected_second = torch.arange(
            6,
            12,
            dtype=torch.float32,
            device=device_type,
        ).reshape(3, 2)

        self.assertTrue(
            torch.equal(outputs[0], expected_first)
        )
        self.assertTrue(
            torch.equal(outputs[1], expected_second)
        )

        flat_storage_ptr = (
            flat.untyped_storage().data_ptr()
        )

        self.assertEqual(
            outputs[0].untyped_storage().data_ptr(),
            flat_storage_ptr,
        )
        self.assertEqual(
            outputs[1].untyped_storage().data_ptr(),
            flat_storage_ptr,
        )

        self.assertEqual(outputs[0].storage_offset(), 0)
        self.assertEqual(outputs[1].storage_offset(), 6)

    def test_unflatten_dense_tensors_with_empty_tensor(self):
        flat = torch.arange(
            6,
            dtype=torch.float32,
            device=device_type,
        )

        tensors = [
            torch.empty(
                (0,),
                dtype=torch.float32,
                device=device_type,
            ),
            torch.empty(
                (2, 3),
                dtype=torch.float32,
                device=device_type,
            ),
        ]

        outputs = torch._C._nn.unflatten_dense_tensors(
            flat,
            tensors,
        )

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)

        self.assertEqual(tuple(outputs[0].shape), (0,))
        self.assertEqual(outputs[0].numel(), 0)
        self.assertEqual(outputs[0].device.type, device_type)

        self.assertEqual(tuple(outputs[1].shape), (2, 3))
        self.assertEqual(outputs[1].device.type, device_type)

        expected = torch.arange(
            6,
            dtype=torch.float32,
            device=device_type,
        ).reshape(2, 3)

        self.assertTrue(
            torch.equal(outputs[1], expected)
        )

        self.assertEqual(
            outputs[1].untyped_storage().data_ptr(),
            flat.untyped_storage().data_ptr(),
        )
        self.assertEqual(outputs[1].storage_offset(), 0)

    def test_unflatten_dense_tensors_autograd(self):
        flat = torch.arange(
            10,
            dtype=torch.float32,
            device=device_type,
            requires_grad=True,
        )

        tensors = [
            torch.empty((2, 3), dtype=torch.float32, device=device_type),
            torch.empty((4,), dtype=torch.float32, device=device_type),
        ]

        outputs = torch._C._nn.unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(outputs), 2)
        self.assertTrue(outputs[0].requires_grad)
        self.assertTrue(outputs[1].requires_grad)
        self.assertEqual(outputs[0].device.type, device_type)
        self.assertEqual(outputs[1].device.type, device_type)

        loss = outputs[0].sum() + (outputs[1] * 2).sum()
        loss.backward()

        expected_grad = torch.tensor(
            [1.0] * 6 + [2.0] * 4,
            dtype=torch.float32,
            device=device_type,
        )

        self.assertIsNotNone(flat.grad)
        self.assertEqual(flat.grad.device.type, device_type)
        self.assertEqual(flat.grad, expected_grad)

    def test_unflatten_dense_tensors_dtype_preservation(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                flat = torch.tensor(
                    [0, 1, 2, 3, 4, 5],
                    dtype=dtype,
                    device=device_type,
                )
                tensors = [
                    torch.empty((2,), dtype=dtype, device=device_type),
                    torch.empty((2, 2), dtype=dtype, device=device_type),
                ]

                outputs = torch._C._nn.unflatten_dense_tensors(flat, tensors)

                self.assertEqual(outputs[0].dtype, dtype)
                self.assertEqual(outputs[1].dtype, dtype)


if __name__ == "__main__":
    run_tests()
