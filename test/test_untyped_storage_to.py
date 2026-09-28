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
Add validation cases for torch.UntypedStorage.to on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API,
   so this file is added.
2. This file validates torch.UntypedStorage.to (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestUntypedStorageTo(TestCase):

    def _assert_storage_values(self, storage, values):
        actual = (
            torch.empty(0, dtype=torch.uint8, device=storage.device)
            .set_(storage)
            .cpu()
        )
        # CPU tensor is the transfer oracle.
        expected = torch.tensor(values, dtype=torch.uint8)
        self.assertEqual(actual, expected)

    def test_to_npu_device_types(self):
        values = [0, 1, 127, 255]
        current_device_index = torch.accelerator.current_device_index()
        # CPU storage is required to validate CPU-to-NPU transfer.
        source = torch.tensor(values, dtype=torch.uint8).untyped_storage()

        for device in (
            device_type,
            torch.device(device_type),
            current_device_index,
        ):
            result = source.to(device=device)
            self.assertIsInstance(result, torch.UntypedStorage)
            self.assertEqual(
                result.device,
                torch.device(device_type, current_device_index),
            )
            self._assert_storage_values(result, values)

    def test_to_npu_non_blocking_options(self):
        values = list(range(8))
        # CPU storages are required to validate host-to-NPU transfer.
        blocking_source = torch.tensor(values, dtype=torch.uint8).untyped_storage()
        pinned_tensor = torch.empty(len(values), dtype=torch.uint8, pin_memory=True)
        pinned_tensor.copy_(torch.tensor(values, dtype=torch.uint8))

        blocking_result = blocking_source.to(device=device_type, non_blocking=False)
        non_blocking_result = pinned_tensor.untyped_storage().to(
            device=device_type, non_blocking=True
        )
        torch.accelerator.synchronize()

        self.assertEqual(blocking_result.device.type, device_type)
        self.assertEqual(non_blocking_result.device.type, device_type)
        self._assert_storage_values(blocking_result, values)
        self._assert_storage_values(non_blocking_result, values)

    def test_to_cpu_non_blocking_options(self):
        values = [3, 5, 8, 13]
        source = (
            torch.tensor(values, dtype=torch.uint8)
            .to(device_type)
            .untyped_storage()
        )

        for non_blocking in (False, True):
            result = source.to(device="cpu", non_blocking=non_blocking)
            if non_blocking:
                torch.accelerator.synchronize()
            self.assertEqual(result.device.type, "cpu")
            self.assertEqual(result.is_pinned(device_type), non_blocking)
            self._assert_storage_values(result, values)

        none_result = source.to(device="cpu", non_blocking=None)
        self.assertEqual(none_result.device.type, "cpu")
        self.assertFalse(none_result.is_pinned(device_type))
        self._assert_storage_values(none_result, values)

    def test_to_same_npu_returns_self(self):
        values = list(range(8))
        storage = (
            torch.tensor(values, dtype=torch.uint8)
            .to(device_type)
            .untyped_storage()
        )

        self.assertIs(storage.to(device=storage.device), storage)

        for device in (
            device_type,
            torch.device(device_type),
            storage.device.index,
        ):
            result = storage.to(device=device)
            self.assertEqual(result.device, storage.device)
            self._assert_storage_values(result, values)

    def test_to_empty_storage(self):
        storage = torch.empty(0, dtype=torch.uint8).to(device_type).untyped_storage()
        cpu_result = storage.to(device="cpu")
        npu_result = cpu_result.to(device=device_type)

        self.assertEqual(cpu_result.nbytes(), 0)
        self.assertEqual(cpu_result.device.type, "cpu")
        self.assertEqual(npu_result.nbytes(), 0)
        self.assertEqual(npu_result.device.type, device_type)

    def test_to_another_npu(self):
        if torch.accelerator.device_count() < 2:
            self.skipTest("Two accelerators are required for cross-device copy")

        values = [2, 4, 6, 8]
        source_device = f"{device_type}:0"
        target_device = f"{device_type}:1"
        storage = (
            torch.tensor(values, dtype=torch.uint8)
            .to(source_device)
            .untyped_storage()
        )
        result = storage.to(device=target_device)

        self.assertEqual(result.device, torch.device(target_device))
        self._assert_storage_values(result, values)

    def test_to_invalid_arguments(self):
        storage = torch.arange(4, dtype=torch.uint8).to(device_type).untyped_storage()

        with self.assertRaises(TypeError):
            storage.to()
        with self.assertRaises(TypeError):
            storage.to(device_type)
        with self.assertRaises(TypeError):
            storage.to(device=None)
        with self.assertRaises(TypeError):
            storage.to(device=[])
        with self.assertRaises(RuntimeError):
            storage.to(device="invalid")
        with self.assertRaises(TypeError):
            storage.to(device="cpu", non_blocking="true")


if __name__ == "__main__":
    run_tests()
