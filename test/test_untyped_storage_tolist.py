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
Add validation cases for torch.UntypedStorage.tolist on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API,
   so this file is added.
2. This file validates torch.UntypedStorage.tolist (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestUntypedStorageToList(TestCase):

    def test_tolist_empty_storage(self):
        storage = torch.empty(0, dtype=torch.uint8).to(device_type).untyped_storage()

        self.assertEqual(storage.tolist(), [])

    def test_tolist_byte_boundaries(self):
        values = [0, 1, 127, 128, 254, 255]
        storage = (
            torch.tensor(values, dtype=torch.uint8)
            .to(device_type)
            .untyped_storage()
        )

        self.assertEqual(storage.tolist(), values)

    def test_tolist_all_byte_values(self):
        values = list(range(256))
        tensor = torch.arange(256, dtype=torch.int32).to(torch.uint8).to(device_type)

        self.assertEqual(tensor.untyped_storage().tolist(), values)

    def test_tolist_storage_slice(self):
        values = [11, 22, 33, 44, 55, 66]
        storage = (
            torch.tensor(values, dtype=torch.uint8)
            .to(device_type)
            .untyped_storage()
        )
        storage_view = storage[1:5]

        self.assertIsInstance(storage_view, torch.UntypedStorage)
        self.assertEqual(storage_view.tolist(), values[1:5])

    def test_tolist_invalid_arguments(self):
        storage = torch.arange(4, dtype=torch.uint8).to(device_type).untyped_storage()

        with self.assertRaises(TypeError):
            storage.tolist(None)
        with self.assertRaises(TypeError):
            storage.tolist(value=None)


if __name__ == "__main__":
    run_tests()
