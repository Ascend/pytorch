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
# Owner(s): ["module: functorch"]
"""
Add validation cases for torch._functorch.vmap tree_flatten and tree_unflatten APIs.

API Introduction:
torch._functorch.vmap.tree_flatten and tree_unflatten are aliases of
torch.utils._pytree.tree_flatten and tree_unflatten. tree_flatten flattens a
nested Python structure into a flat list of leaves and a TreeSpec describing
the structure. tree_unflatten reconstructs the original structure from a flat
list and a TreeSpec.

Test Design:
1. PyTorch community lacks sufficient and direct API validations for these APIs,
   so this file is added.
2. This file validates torch._functorch.vmap.tree_flatten and tree_unflatten
   in the following scenarios:
   - flatten list, dict, and nested structures
   - flatten structures containing tensors
   - unflatten list and dict back to original structure
   - roundtrip (flatten then unflatten) for nested structures and tensors
"""
import torch
from torch._functorch.vmap import tree_flatten, tree_unflatten
from torch.testing._internal.common_utils import TestCase, run_tests
acc = torch.accelerator.current_accelerator()
if acc is None:
    raise RuntimeError("No available accelerator. This test requires an NPU/GPU.")
device_type = acc.type


class TestVmapPytreeAPIs(TestCase):
    """Tests for torch._functorch.vmap.tree_flatten and tree_unflatten APIs."""

    def test_tree_flatten_list(self):
        data = [1, 2, 3]
        flat, spec = tree_flatten(data)
        self.assertEqual(flat, [1, 2, 3])

    def test_tree_flatten_dict(self):
        data = {'a': 1, 'b': 2}
        flat, spec = tree_flatten(data)
        self.assertEqual(len(flat), 2)
        self.assertIn(1, flat)
        self.assertIn(2, flat)

    def test_tree_flatten_nested(self):
        data = {'a': [1, 2], 'b': {'c': 3}}
        flat, spec = tree_flatten(data)
        self.assertEqual(len(flat), 3)
        self.assertIn(1, flat)
        self.assertIn(2, flat)
        self.assertIn(3, flat)

    def test_tree_flatten_with_tensor(self):
        t = torch.tensor([1.0, 2.0]).to(device_type)
        data = [t, t]
        flat, spec = tree_flatten(data)
        self.assertEqual(len(flat), 2)
        self.assertTrue(torch.equal(flat[0], t))
        self.assertTrue(torch.equal(flat[1], t))

    def test_tree_unflatten_list(self):
        data = [1, 2, 3]
        flat, spec = tree_flatten(data)
        restored = tree_unflatten(flat, spec)
        self.assertEqual(restored, data)

    def test_tree_unflatten_dict(self):
        data = {'a': 1, 'b': 2}
        flat, spec = tree_flatten(data)
        restored = tree_unflatten(flat, spec)
        self.assertEqual(restored, data)

    def test_tree_flatten_unflatten_roundtrip_nested(self):
        data = {'x': [1, 2], 'y': {'z': 3}}
        flat, spec = tree_flatten(data)
        restored = tree_unflatten(flat, spec)
        self.assertEqual(restored, data)

    def test_tree_flatten_unflatten_roundtrip_with_tensors(self):
        t1 = torch.tensor([1.0]).to(device_type)
        t2 = torch.tensor([2.0]).to(device_type)
        data = {'a': t1, 'b': [t2]}
        flat, spec = tree_flatten(data)
        restored = tree_unflatten(flat, spec)
        self.assertTrue(torch.equal(restored['a'], t1))
        self.assertTrue(torch.equal(restored['b'][0], t2))


if __name__ == '__main__':
    run_tests()
