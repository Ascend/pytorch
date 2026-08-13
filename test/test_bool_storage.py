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
Add consistency validation cases for torch.BoolStorage / torch_npu.npu.BoolStorage
on NPU (#2955).

The PyTorch community lacks a dedicated NPU consistency test for the BoolStorage
storage class. Upstream only checks ``torch.BoolStorage().element_size()`` in
``test_torch.py::test_element_size`` (a CPU-only property check), and the existing
``test/npu/test_storage.py`` exercises storage via tensor ``.storage()`` rather than
the BoolStorage class itself. This file validates the BoolStorage class behavior
(construction from size / sequence, empty object, element_size, indexing, out-of-
bounds access, fill_, tolist, roundtrip) on both CPU and NPU, and asserts an NPU
bool tensor's ``.storage()`` is a ``torch_npu.npu.BoolStorage`` instance with the
expected dtype and data consistency.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu.npu as npu


class TestBoolStorage(TestCase):

    # ---------------- CPU: torch.BoolStorage ----------------
    def test_cpu_bool_storage_basic(self):
        # Core CPU BoolStorage behavior (construction by size, indexing, fill_).
        s = torch.BoolStorage(3)
        s[0] = True
        s[1] = False
        s[2] = True
        self.assertEqual(s.tolist(), [True, False, True])
        self.assertEqual(s.size(), 3)
        self.assertEqual(s.element_size(), 1)
        self.assertEqual(torch.BoolStorage().element_size(), 1)

        # fill_ then tolist
        s.fill_(False)
        self.assertEqual(s.tolist(), [False, False, False])

        # roundtrip through torch.BoolTensor (CPU storage -> CPU tensor)
        t = torch.BoolTensor(s)
        self.assertEqual(t.dtype, torch.bool)
        self.assertEqual(t.tolist(), [False, False, False])

    def test_cpu_bool_storage_from_sequence(self):
        # Minimal sequence construction (single element).
        s1 = torch.BoolStorage([True])
        self.assertEqual(s1.tolist(), [True])
        self.assertEqual(s1.size(), 1)

        # Sequence construction (multiple elements).
        s2 = torch.BoolStorage([True, False, True])
        self.assertEqual(s2.tolist(), [True, False, True])
        self.assertEqual(s2.size(), 3)

    def test_cpu_bool_storage_empty(self):
        # Empty object construction.
        s = torch.BoolStorage()
        self.assertEqual(s.size(), 0)
        self.assertEqual(s.tolist(), [])

        # Explicit zero-size construction.
        s0 = torch.BoolStorage(0)
        self.assertEqual(s0.size(), 0)
        self.assertEqual(s0.tolist(), [])

    def test_cpu_bool_storage_out_of_bounds(self):
        s = torch.BoolStorage(3)
        # Read beyond the end raises IndexError.
        with self.assertRaises(IndexError):
            _ = s[3]
        # Write beyond the end raises IndexError.
        with self.assertRaises(IndexError):
            s[3] = True

    # ---------------- NPU: torch_npu.npu.BoolStorage ----------------
    def test_npu_bool_storage_basic(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

        self.assertTrue(hasattr(npu, "BoolStorage"))
        ns = npu.BoolStorage(4)
        ns[0] = True
        ns[1] = False
        ns[2] = True
        ns[3] = False
        self.assertEqual(ns.tolist(), [True, False, True, False])
        self.assertEqual(ns.dtype, torch.bool)
        self.assertEqual(ns.element_size(), 1)
        self.assertEqual(ns.size(), 4)

        ns.fill_(True)
        self.assertEqual(ns.tolist(), [True, True, True, True])

    def test_npu_bool_storage_from_sequence(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

        # Minimal sequence construction on NPU.
        ns = npu.BoolStorage([True, False])
        self.assertEqual(ns.tolist(), [True, False])
        self.assertEqual(ns.size(), 2)

    def test_npu_bool_storage_empty(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

        ns = npu.BoolStorage()
        self.assertEqual(ns.size(), 0)
        self.assertEqual(ns.tolist(), [])

    def test_npu_bool_storage_out_of_bounds(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

        ns = npu.BoolStorage(2)
        with self.assertRaises(IndexError):
            _ = ns[2]
        with self.assertRaises(IndexError):
            ns[2] = True

    def test_npu_tensor_storage_consistency(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

        # An NPU bool tensor's storage is a torch_npu.npu.BoolStorage instance,
        # with matching dtype and data consistency.
        x = torch.tensor([True, False, True], device="npu")
        self.assertIsInstance(x.storage(), npu.BoolStorage)
        self.assertEqual(x.storage().dtype, torch.bool)
        self.assertEqual(x.storage().tolist(), [True, False, True])


if __name__ == "__main__":
    run_tests()
