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
Add consistency validation cases for torch.Tensor.map_.

This file adds self-contained, focused validation for torch.Tensor.map_ as requested by the
Ascend for PyTorch API consistency task (#2720). Extendable.

PyTorch community test: test/test_torch.py::test_broadcast includes "map" in its
parametrized fn list, but the test is explicitly skipped on CUDA devices
("map and map2 are not implemented on CUDA tensors"). NPU follows the same
behavior -- map_ is CPU-only, so functional validation is done on CPU tensors.
NPU tensors are verified to raise the expected error.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTensorMap(TestCase):
    def test_map_applies_callable(self):
        # map_ is CPU-only in PyTorch community (same as CUDA)
        dst = torch.zeros(3)
        src = torch.tensor([1.0, 2.0, 3.0])
        dst.map_(src, lambda d, s: s * 2)
        self.assertEqual(dst, torch.tensor([2.0, 4.0, 6.0]))

    def test_map_uses_destination_values(self):
        # map_ is CPU-only in PyTorch community (same as CUDA)
        dst = torch.tensor([10.0, 20.0])
        src = torch.tensor([1.0, 2.0])
        dst.map_(src, lambda d, s: d + s)
        self.assertEqual(dst, torch.tensor([11.0, 22.0]))

    def test_map_raises_on_npu_tensor(self):
        # Verify that map_ raises TypeError with the expected message on NPU tensors
        dst = torch.zeros(3).npu()
        src = torch.tensor([1.0, 2.0, 3.0]).npu()
        with self.assertRaises(TypeError) as ctx:
            dst.map_(src, lambda d, s: s * 2)
        self.assertIn("map_ is only implemented on CPU", str(ctx.exception))


if __name__ == "__main__":
    run_tests()
