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
Add validation cases for torch.TypedStorage and torch.UntypedStorage on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs, so this file is added.
2. This file validates torch.UntypedStorage.fill_, torch.UntypedStorage.float, torch.TypedStorage.int, and torch.UntypedStorage (extendable).
"""
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestUntypedStorageFillNPU(TestCase):
    def test_fill_returns_self(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        result = storage.fill_(7)
        self.assertIs(result, storage)

    def test_fill_byte_value(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(7)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [7, 7, 7, 7])

    def test_fill_zero(self):
        storage = torch.full((4,), 5, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(0)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [0, 0, 0, 0])

    def test_fill_empty(self):
        storage = torch.zeros(0, dtype=torch.uint8, device=device_type).untyped_storage()
        result = storage.fill_(3)
        self.assertIs(result, storage)
        self.assertEqual(storage.nbytes(), 0)


    def test_fill_min_byte_value(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(0)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [0, 0, 0, 0])

    def test_fill_max_byte_value(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(255)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [255, 255, 255, 255])

    def test_fill_mid_byte_value(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(127)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [127, 127, 127, 127])

    def test_fill_out_of_range_high(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(256)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [0, 0, 0, 0])

    def test_fill_negative_value(self):
        storage = torch.zeros(4, dtype=torch.uint8, device=device_type).untyped_storage()
        storage.fill_(-1)
        view = torch.tensor([], dtype=torch.uint8, device=device_type).set_(storage)
        self.assertEqual(view.tolist(), [255, 255, 255, 255])


class TestUntypedStorageFloatNPU(TestCase):
    def test_float_returns_typed_storage(self):
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device_type)
        storage = t.untyped_storage()
        result = storage.float()
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, device_type)

    def test_float_length_equals_nbytes(self):
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device_type)
        storage = t.untyped_storage()
        result = storage.float()
        self.assertEqual(result.size(), storage.nbytes())

    def test_float_data_correctness(self):
        t = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=device_type)
        storage = t.untyped_storage()
        result = storage.float()
        view = torch.tensor([], dtype=torch.float32, device=device_type).set_(result)
        self.assertEqual(view.tolist(), [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0])

    def test_float_empty(self):
        t = torch.zeros(0, dtype=torch.int32, device=device_type)
        storage = t.untyped_storage()
        result = storage.float()
        self.assertEqual(result.size(), 0)


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


if __name__ == "__main__":
    run_tests()
