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
Add validation cases for torch.Tensor.unsqueeze_ on NPU:
1. PyTorch community tests cover basic inplace view behavior but not all dimension boundaries.
2. This file validates torch.Tensor.unsqueeze_ across shapes, aliases, autograd, and errors.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTensorUnsqueezeAPIs(TestCase):

    def test_unsqueeze_dimensions(self):
        device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
        source = torch.arange(6, dtype=torch.float32, device=device_type).reshape(2, 3)
        cases = (
            (0, (1, 2, 3)),
            (1, (2, 1, 3)),
            (2, (2, 3, 1)),
            (-1, (2, 3, 1)),
            (-2, (2, 1, 3)),
            (-3, (1, 2, 3)),
        )

        for dim, expected_shape in cases:
            value = source.clone()
            result = value.unsqueeze_(dim)
            self.assertIs(result, value)
            self.assertEqual(result.shape, expected_shape)
            self.assertEqual(result, source.reshape(expected_shape))

        scalar = torch.tensor(1.0, device=device_type)
        for dim in (0, -1):
            value = scalar.clone()
            result = value.unsqueeze_(dim)
            self.assertEqual(result.shape, (1,))
            self.assertEqual(result, torch.ones(1, device=device_type))

    def test_unsqueeze_view_alias(self):
        device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
        base = torch.arange(12, dtype=torch.float32, device=device_type).reshape(3, 4)
        view = base[:, ::2]
        result = view.unsqueeze_(-1)

        self.assertEqual(result.shape, (3, 2, 1))
        result[1, 1, 0] = -5
        self.assertEqual(base[1, 2], torch.tensor(-5.0, device=device_type))

    def test_unsqueeze_autograd(self):
        device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
        leaf = torch.ones(2, 3, device=device_type, requires_grad=True)
        with self.assertRaises(RuntimeError):
            leaf.unsqueeze_(0)

        value = leaf * 2
        result = value.unsqueeze_(-1)
        result.sum().backward()
        self.assertEqual(leaf.grad, torch.full_like(leaf, 2, device=device_type))

    def test_unsqueeze_invalid_dimensions(self):
        device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
        source = torch.ones(2, 3, device=device_type)
        for dim in (-4, 3):
            with self.assertRaises(IndexError):
                source.clone().unsqueeze_(dim)


if __name__ == "__main__":
    run_tests()
