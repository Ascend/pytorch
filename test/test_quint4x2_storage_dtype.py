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
Add consistency validation cases for torch.QUInt4x2Storage.dtype on NPU (#3536).

``torch.QUInt4x2Storage`` is the storage class backing the ``torch.quint4x2``
quantized dtype. Its ``dtype`` attribute must report ``torch.quint4x2``
consistently. This file validates the dtype value both on CPU and in an
NPU-enabled environment, and asserts that the reported dtype is the expected
quantized dtype (not just a truthy value).

Note: NPU does not currently register a dedicated ``torch.npu.QUInt4x2Storage``
storage class (the curated NPU storage list only covers the numeric/bool/
bfloat16 storages), so the device-independent ``torch.QUInt4x2Storage.dtype``
class attribute is the canonical value validated here; should NPU later expose
a ``torch.npu.QUInt4x2Storage`` class, the NPU test additionally checks it.

This file can be extended with more storage-dtype consistency cases for other
quantized storage classes (e.g. QUInt8Storage, QInt8Storage) following the same
pattern.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch_npu  # noqa: F401  # imported for side effects (loads the NPU backend)


class TestQUInt4x2StorageDtype(TestCase):

    # ---------------- CPU: torch.QUInt4x2Storage.dtype ----------------
    def test_cpu_quint4x2_storage_dtype(self):
        # The QUInt4x2Storage class must be provided by this PyTorch build.
        if not hasattr(torch, "QUInt4x2Storage"):
            self.skipTest(
                "torch.QUInt4x2Storage is not available in this PyTorch build"
            )
        # The QUInt4x2Storage class must report the correct quantized dtype.
        self.assertIs(torch.QUInt4x2Storage.dtype, torch.quint4x2)
        # quint4x2 is a quantized dtype. NOTE: torch.dtype objects do not expose
        # an ``is_quantized`` attribute (that lives on torch.Tensor), so we
        # verify membership in the known quantized-dtype set instead.
        quantized_dtypes = (
            torch.quint8, torch.qint8, torch.qint32, torch.quint4x2,
        )
        self.assertIn(torch.QUInt4x2Storage.dtype, quantized_dtypes)
        # The reported dtype must be specifically quint4x2, not another
        # quantized dtype (boundary: distinguish from quint8 / qint8).
        self.assertIsNot(torch.QUInt4x2Storage.dtype, torch.quint8)
        self.assertIsNot(torch.QUInt4x2Storage.dtype, torch.qint8)
        # The instance dtype must be consistent with the class attribute dtype
        # (the canonical value validated above).
        inst = torch.QUInt4x2Storage()
        self.assertIs(inst.dtype, torch.QUInt4x2Storage.dtype)
        self.assertIs(inst.dtype, torch.quint4x2)

    # ---------------- NPU: consistency in an NPU-enabled environment ----------------
    def test_npu_quint4x2_storage_dtype(self):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            self.skipTest("NPU not available")
        # QUInt4x2Storage must be provided by this PyTorch build even on NPU;
        # guard it so a build without quantized storages skips gracefully
        # instead of raising AttributeError (consistent with the CPU test).
        if not hasattr(torch, "QUInt4x2Storage"):
            self.skipTest(
                "torch.QUInt4x2Storage is not available in this PyTorch build"
            )

        # The dtype attribute is device-independent: it must remain
        # torch.quint4x2 even in an NPU-enabled environment.
        self.assertIs(torch.QUInt4x2Storage.dtype, torch.quint4x2)

        # If NPU later registers a dedicated quint4x2 storage class, verify it
        # reports the same dtype. Guarded so the test stays valid whether or not
        # the NPU storage class is currently exposed.
        if hasattr(torch.npu, "QUInt4x2Storage"):
            self.assertIs(torch.npu.QUInt4x2Storage.dtype, torch.quint4x2)
            ns = torch.npu.QUInt4x2Storage()
            self.assertIs(ns.dtype, torch.quint4x2)
            self.assertEqual(ns.size(), 0)


if __name__ == "__main__":
    run_tests()
