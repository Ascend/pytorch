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
Add validation cases for torch.TypedStorage.from_file on NPU:
1. PyTorch community tests shared mappings but lacks private and invalid-input cases.
2. This file validates typed file mappings and transfer of mapped data to NPU.
"""

import os
import struct
import tempfile

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTypedStorageFromFile(TestCase):

    def test_shared_mapping_and_npu_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "shared.bin")
            storage = torch.FloatStorage.from_file(filename, True, 4)
            for index, value in enumerate((1.25, 2.5, 3.75, 4.5)):
                storage[index] = value

            # The file mapping itself must remain on CPU.
            self.assertEqual(storage.device.type, "cpu")
            self.assertEqual(os.path.getsize(filename), 16)
            other = torch.FloatStorage.from_file(
                filename=filename, shared=True, size=4
            )
            self.assertEqual(list(other), [1.25, 2.5, 3.75, 4.5])

            mapped = torch.FloatTensor(storage)
            npu_tensor = mapped.to(device_type)
            self.assertEqual(npu_tensor.device.type, device_type)
            self.assertEqual(npu_tensor.cpu().tolist(), [1.25, 2.5, 3.75, 4.5])

            npu_tensor.fill_(6.5)
            mapped.copy_(npu_tensor.cpu())
            self.assertEqual(list(other), [6.5] * 4)
            with open(filename, "rb") as file_handle:
                self.assertEqual(
                    struct.unpack("=4f", file_handle.read()), (6.5,) * 4
                )

    def test_private_mapping_does_not_change_file(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "private.bin")
            original = struct.pack("=4f", 1.0, 2.0, 3.0, 4.0)
            with open(filename, "wb") as file_handle:
                file_handle.write(original)

            storage = torch.FloatStorage.from_file(filename, False, 4)
            self.assertEqual(storage.device.type, "cpu")
            self.assertEqual(list(storage), [1.0, 2.0, 3.0, 4.0])
            storage[0] = 9.0
            self.assertEqual(storage[0], 9.0)
            with open(filename, "rb") as file_handle:
                self.assertEqual(file_handle.read(), original)

    def test_unicode_filename_and_byte_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "映射文件.bin")
            storage = torch.ByteStorage.from_file(filename, True, 3)
            self.assertEqual(storage.dtype, torch.uint8)
            self.assertEqual(len(storage), 3)
            self.assertEqual(os.path.getsize(filename), 3)

            storage[0] = 7
            storage[1] = 128
            storage[2] = 255
            other = torch.ByteStorage.from_file(filename, True, 3)
            self.assertEqual(list(other), [7, 128, 255])
            with open(filename, "rb") as file_handle:
                self.assertEqual(file_handle.read(), bytes([7, 128, 255]))

    def test_zero_size_and_integer_shared_values(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "data.bin")
            with open(filename, "wb") as file_handle:
                file_handle.write(b"\0" * 16)

            for shared in (False, True):
                with self.subTest(shared=shared):
                    storage = torch.FloatStorage.from_file(filename, shared, 0)
                    self.assertEqual(len(storage), 0)
                    self.assertEqual(storage.device.type, "cpu")

            for shared in (0, 1):
                with self.subTest(shared=shared):
                    storage = torch.FloatStorage.from_file(filename, shared, 4)
                    self.assertEqual(len(storage), 4)

    def test_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "data.bin")
            with open(filename, "wb") as file_handle:
                file_handle.write(b"\0" * 16)

            with self.assertRaisesRegex(RuntimeError, "derived classes"):
                torch.TypedStorage.from_file(filename, True, 4)
            for kwargs in (
                {"shared": False, "size": 4},
                {"filename": filename, "size": 4},
                {"filename": filename, "shared": False},
            ):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(TypeError):
                        torch.FloatStorage.from_file(**kwargs)
            with self.assertRaises(TypeError):
                torch.FloatStorage.from_file(None, True, 4)
            with self.assertRaises(TypeError):
                torch.FloatStorage.from_file(123, False, 4)
            with self.assertRaises(TypeError):
                torch.FloatStorage.from_file(filename, None, 4)
            with self.assertRaises(TypeError):
                torch.FloatStorage.from_file(filename, False, None)
            with self.assertRaises(TypeError):
                torch.FloatStorage.from_file(filename, False, 1.5)
            with self.assertRaises(RuntimeError):
                torch.FloatStorage.from_file("", False, 4)
            with self.assertRaises(RuntimeError):
                torch.FloatStorage.from_file(filename, False, -1)
            with self.assertRaisesRegex(RuntimeError, "smaller than"):
                torch.FloatStorage.from_file(filename, False, 5)
            with self.assertRaises(RuntimeError):
                torch.FloatStorage.from_file(
                    os.path.join(directory, "missing.bin"), False, 4
                )


if __name__ == "__main__":
    run_tests()
