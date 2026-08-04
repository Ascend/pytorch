#!/usr/bin/env python3
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
# Owner(s): ["module: library"]

"""Add validation cases for torch._refs._maybe_broadcast on NPU:
1. PyTorch community lacks sufficient and direct API validations for
   torch._refs._maybe_broadcast, so this file is added.
2. This file validates the broadcasting behavior of _maybe_broadcast for
   TensorLike inputs, including same-shape, compatible-shape, CPU-scalar
   preservation/expansion, and error cases (extendable).
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestLibraryRefs(TestCase):
    """Test torch._refs._maybe_broadcast on NPU and CPU scalar tensors."""

    def test_maybe_broadcast_same_shape_npu(self):
        """Tensors with the same NPU shape are returned unchanged."""
        x = torch.empty(2, 3).to(device_type)
        y = torch.empty(2, 3).to(device_type)
        bx, by = torch._refs._maybe_broadcast(x, y)
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(by.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertEqual(by.device, y.device)
        self.assertIs(bx, x)
        self.assertIs(by, y)

    def test_maybe_broadcast_different_shapes_npu(self):
        """NPU tensors with compatible shapes are expanded to the common shape."""
        x = torch.empty(1, 3).to(device_type)
        y = torch.empty(2, 1).to(device_type)
        bx, by = torch._refs._maybe_broadcast(x, y)
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(by.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertEqual(by.device, y.device)

    def test_maybe_broadcast_number(self):
        """A Number argument is returned as-is."""
        x = torch.empty(2, 3).to(device_type)
        number = 2.0
        bx, bnumber = torch._refs._maybe_broadcast(x, number)
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertEqual(bnumber, number)

    def test_maybe_broadcast_none(self):
        """A None argument is returned as None."""
        x = torch.empty(2, 3).to(device_type)
        bx, bnone = torch._refs._maybe_broadcast(x, None)
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertIsNone(bnone)

    def test_maybe_broadcast_cpu_scalar_preserved(self):
        """CPU scalar tensors are preserved when preserve_cpu_scalar_tensors is True."""
        x = torch.empty(2, 3).to(device_type)
        scalar = torch.tensor(1.0)  # CPU scalar, intentionally not on NPU
        bx, bscalar = torch._refs._maybe_broadcast(
            x, scalar, preserve_cpu_scalar_tensors=True
        )
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertEqual(bscalar.shape, ())
        self.assertEqual(bscalar.device, torch.device("cpu"))
        self.assertIs(bscalar, scalar)

    def test_maybe_broadcast_cpu_scalar_expanded(self):
        """CPU scalar tensors are expanded when preserve_cpu_scalar_tensors is False."""
        x = torch.empty(2, 3).to(device_type)
        scalar = torch.tensor(1.0)  # CPU scalar, intentionally not on NPU
        bx, bscalar = torch._refs._maybe_broadcast(
            x, scalar, preserve_cpu_scalar_tensors=False
        )
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertEqual(bscalar.shape, (2, 3))
        self.assertEqual(bscalar.device, torch.device("cpu"))

    def test_maybe_broadcast_scalar_expanded(self):
        """A 0-dim scalar tensor is broadcast when CPU preservation is disabled."""
        x = torch.empty(2, 3).to(device_type)
        scalar = torch.tensor(1.0).to(device_type)
        bx, bscalar = torch._refs._maybe_broadcast(
            x, scalar, preserve_cpu_scalar_tensors=False
        )
        self.assertEqual(bx.shape, (2, 3))
        self.assertEqual(bx.device, x.device)
        self.assertEqual(bscalar.shape, (2, 3))
        self.assertEqual(bscalar.device, x.device)

    def test_maybe_broadcast_incompatible_shapes(self):
        """Incompatible tensor shapes raise RuntimeError."""
        x = torch.empty(2, 3).to(device_type)
        y = torch.empty(3, 2).to(device_type)
        with self.assertRaises(RuntimeError):
            torch._refs._maybe_broadcast(x, y)


if __name__ == "__main__":
    run_tests()
