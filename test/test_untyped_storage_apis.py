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
# Owner(s): ["module: storage"]
"""
Add validation cases for torch.UntypedStorage APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.UntypedStorage.is_cuda, torch.UntypedStorage.is_pinned (extendable).
"""

import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

_acc = torch.accelerator.current_accelerator()
HAS_NPU = _acc is not None and _acc.type == "npu"
device_type = _acc.type if HAS_NPU else "cpu"


class TestUntypedStorageIsCuda(TestCase):

    def test_is_cuda_on_cpu(self):
        storage = torch.UntypedStorage(10)
        self.assertFalse(storage.is_cuda)

    @unittest.skipUnless(HAS_NPU, "NPU not available")
    def test_is_cuda_on_npu(self):
        storage = torch.UntypedStorage(10, device=device_type)
        self.assertFalse(storage.is_cuda)


class TestUntypedStorageIsPinned(TestCase):

    def test_is_pinned_on_cpu(self):
        storage = torch.UntypedStorage(10)
        self.assertFalse(storage.is_pinned())

    @unittest.skipUnless(HAS_NPU, "NPU not available")
    def test_is_pinned_on_npu(self):
        storage = torch.UntypedStorage(10, device=device_type)
        self.assertFalse(storage.is_pinned())


if __name__ == "__main__":
    run_tests()
