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
Add validation cases for torch.is_nonzero on NPU:
1. Upstream function-form coverage in test/test_mps.py only checks four
   single-element positive paths and does not cover empty/multi-element
   RuntimeError, None TypeError, or complex zero/non-zero behavior.
2. This file validates torch.is_nonzero core behavior on accelerator tensors
   (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestIsNonzero(TestCase):
    def test_is_nonzero_single_element_false(self):
        self.assertFalse(torch.is_nonzero(torch.tensor([0.0]).to(device_type)))
        self.assertFalse(torch.is_nonzero(torch.tensor([0]).to(device_type)))
        self.assertFalse(torch.is_nonzero(torch.tensor([False]).to(device_type)))
        self.assertFalse(torch.is_nonzero(torch.tensor(0).to(device_type)))
        self.assertFalse(torch.is_nonzero(torch.tensor(0.0).to(device_type)))
        self.assertFalse(torch.is_nonzero(torch.tensor(False).to(device_type)))

    def test_is_nonzero_single_element_true(self):
        self.assertTrue(torch.is_nonzero(torch.tensor([1.5]).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor([3]).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor([True]).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor([[1]]).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor(1).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor(-0.1).to(device_type)))

    def test_is_nonzero_complex(self):
        self.assertFalse(torch.is_nonzero(torch.tensor([0 + 0j]).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor([0 + 0.1j]).to(device_type)))
        self.assertFalse(torch.is_nonzero(torch.tensor(0 + 0j).to(device_type)))
        self.assertTrue(torch.is_nonzero(torch.tensor(0 + 0.1j).to(device_type)))

    def test_is_nonzero_empty_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no values"):
            torch.is_nonzero(torch.tensor([]).to(device_type))

    def test_is_nonzero_multi_element_raises(self):
        with self.assertRaisesRegex(RuntimeError, "more than one value"):
            torch.is_nonzero(torch.tensor([1, 3, 5]).to(device_type))
        with self.assertRaisesRegex(RuntimeError, "more than one value"):
            torch.is_nonzero(torch.tensor([0, 0]).to(device_type))

    def test_is_nonzero_none_raises(self):
        with self.assertRaisesRegex(TypeError, r"must be Tensor"):
            torch.is_nonzero(None)


if __name__ == "__main__":
    run_tests()
