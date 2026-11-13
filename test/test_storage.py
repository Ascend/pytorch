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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.TypedStorage and torch.UntypedStorage APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.TypedStorage.int, torch.UntypedStorage,
   torch.TypedStorage.byte and torch.TypedStorage.cpu (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTypedStorage(TestCase):

    def test_int_npu(self):
        # Validate TypedStorage with int for boundary sizes, consistency, and scalar read/write operations.
        for size in [0, 4, 100]:
            cpu_base_storage = torch.TypedStorage(size, dtype=torch.float32, device="cpu")
            npu_base_storage = torch.TypedStorage(size, dtype=torch.float32, device=device_type)
            cpu_storage = cpu_base_storage.int()
            npu_storage = npu_base_storage.int()
            self.assertEqual(npu_storage.size(), cpu_storage.size())
            self.assertEqual(npu_storage.dtype, torch.int)
            self.assertEqual(npu_storage.dtype, cpu_storage.dtype)
            if size > 0:
                cpu_storage.fill_(1)
                npu_storage.copy_(cpu_storage)
                npu_storage[0] = 5
                self.assertEqual(npu_storage[0], 5)


class TestUntypedStorage(TestCase):

    def test_untyped_storage_npu(self):
        # Validate UntypedStorage for boundary sizes, CPU/NPU consistency, and basic data assignment.
        for size in [0, 10, 100]:
            cpu_storage = torch.UntypedStorage(size, device="cpu")
            npu_storage = torch.UntypedStorage(size, device=device_type)
            self.assertEqual(npu_storage.size(), cpu_storage.size())
            if size > 0:
                cpu_storage[0] = 1
                npu_storage.copy_(cpu_storage)
                self.assertEqual(npu_storage[0], 1)


class TestTypedStorageByteNPU(TestCase):
    def test_byte_returns_typed_storage(self):
        storage = torch.tensor([1, 2, 3], dtype=torch.int32, device=device_type).storage()
        result = storage.byte()
        self.assertIsInstance(result, torch.storage.TypedStorage)
        self.assertEqual(result.dtype, torch.uint8)

    def test_byte_device_npu(self):
        storage = torch.tensor([1, 2, 3], dtype=torch.int32, device=device_type).storage()
        result = storage.byte()
        self.assertEqual(result.device.type, device_type)

    def test_byte_data_correctness(self):
        storage = torch.tensor([1, 2, 3], dtype=torch.int32, device=device_type).storage()
        result = storage.byte()
        self.assertEqual(result.tolist(), [1, 2, 3])

    def test_byte_multiple_dtypes(self):
        for dtype in [torch.int8, torch.int32, torch.float32, torch.float16]:
            storage = torch.tensor([1, 2, 3], dtype=dtype, device=device_type).storage()
            result = storage.byte()
            self.assertEqual(result.dtype, torch.uint8)
            self.assertEqual(result.size(), 3)

    def test_byte_does_not_modify_original(self):
        storage = torch.tensor([5, 6, 7], dtype=torch.int32, device=device_type).storage()
        result = storage.byte()
        self.assertEqual(storage.tolist(), [5, 6, 7])
        self.assertEqual(result.tolist(), [5, 6, 7])


class TestTypedStorageCpuNPU(TestCase):
    def test_cpu_returns_typed_storage(self):
        storage = torch.tensor([1, 2, 3], dtype=torch.int32, device=device_type).storage()
        result = storage.cpu()
        self.assertIsInstance(result, torch.storage.TypedStorage)
        self.assertEqual(result.device.type, "cpu")

    def test_cpu_dtype_preserved(self):
        storage = torch.tensor([1, 2, 3], dtype=torch.float32, device=device_type).storage()
        result = storage.cpu()
        self.assertEqual(result.dtype, torch.float32)

    def test_cpu_data_correctness(self):
        storage = torch.tensor([1, 2, 3], dtype=torch.int32, device=device_type).storage()
        result = storage.cpu()
        self.assertEqual(result.tolist(), [1, 2, 3])

    def test_cpu_size_preserved(self):
        storage = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device_type).storage()
        result = storage.cpu()
        self.assertEqual(result.size(), 5)


if __name__ == "__main__":
    run_tests()
