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
Add validation cases for torch.UntypedStorage.half on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API,
   so this file is added.
2. This file validates torch.UntypedStorage.half, which reinterprets the raw
   bytes of the storage as uint8 elements and casts them to torch.float16
   values (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def _bytes_of(tensor):
    return list(tensor.contiguous().view(torch.uint8).cpu().tolist())


class TestUntypedStorageHalf(TestCase):

    def _assert_half_bytes(self, storage, tensor):
        result = storage.half()

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.device.type, storage.device.type)
        # element count of the result equals the byte count of the source
        values = torch.empty(0, dtype=torch.float16, device=result.device).set_(result)
        self.assertEqual(values.numel(), storage.nbytes())
        expected = torch.tensor(_bytes_of(tensor), dtype=torch.float16)
        self.assertEqual(values.cpu(), expected)

    def test_half_empty_storage(self):
        storage = torch.empty(0, dtype=torch.uint8).to(device_type).untyped_storage()

        result = storage.half()

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(
            torch.empty(0, dtype=torch.float16, device=result.device)
            .set_(result)
            .numel(),
            0,
        )

    def test_half_from_float32(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device_type)
        storage = tensor.untyped_storage()

        self._assert_half_bytes(storage, tensor)

    def test_half_from_int32(self):
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32).to(device_type)
        storage = tensor.untyped_storage()

        self._assert_half_bytes(storage, tensor)

    def test_half_byte_boundaries(self):
        tensor = torch.tensor([0, 1, 127, 128, 254, 255], dtype=torch.uint8).to(device_type)
        storage = tensor.untyped_storage()

        self._assert_half_bytes(storage, tensor)

    def test_half_does_not_modify_source(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device_type)
        storage = tensor.untyped_storage()
        expected_bytes = _bytes_of(tensor)

        result = storage.half()

        self.assertEqual(_bytes_of(tensor), expected_bytes)

    def test_half_cpu_storage(self):
        tensor = torch.tensor([0.5, 1.5, -3.5], dtype=torch.float32)
        storage = tensor.untyped_storage()

        self._assert_half_bytes(storage, tensor)


if __name__ == "__main__":
    run_tests()
