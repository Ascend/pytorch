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
Add validation cases for torch.UntypedStorage APIs on NPU.

This file validates torch.UntypedStorage.filename,
torch.UntypedStorage.hpu, torch.UntypedStorage.nbytes,
and torch.UntypedStorage.new on CPU and NPU.
"""

import os
import tempfile

import torch

from torch_npu.testing.testcase import TestCase, run_tests


class TestUntypedStorageAPIs(TestCase):

    def test_untyped_storage_filename(self):
        # Scenario 1: Regular CPU storage is not associated with a memory-mapped file
        cpu_storage = torch.empty(
            8,
            dtype=torch.float32,
            device="cpu",
        ).untyped_storage()

        self.assertEqual(cpu_storage.device.type, "cpu")
        self.assertIsNone(cpu_storage.filename)

        # Scenarios 2 and 3: Shared and private file-backed storage mappings
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(
                temp_dir,
                "untyped_storage.bin",
            )

            shared_storage = torch.UntypedStorage.from_file(
                file_path,
                shared=True,
                nbytes=32,
            )

            self.assertEqual(shared_storage.device.type, "cpu")
            self.assertEqual(shared_storage.nbytes(), 32)
            self.assertIsInstance(
                shared_storage.filename,
                str,
            )
            self.assertEqual(
                os.path.realpath(shared_storage.filename),
                os.path.realpath(file_path),
            )

            # Release the shared mapping before creating the private mapping
            del shared_storage

            private_storage = torch.UntypedStorage.from_file(
                file_path,
                shared=False,
                nbytes=32,
            )

            self.assertEqual(private_storage.device.type, "cpu")
            self.assertEqual(private_storage.nbytes(), 32)
            self.assertIsNone(private_storage.filename)

            del private_storage

        # Scenario 4: NPU storage is not associated with a CPU memory-mapped file
        npu_storage = torch.empty(
            8,
            dtype=torch.float32,
            device="npu",
        ).untyped_storage()

        self.assertEqual(npu_storage.device.type, "npu")
        self.assertIsNone(npu_storage.filename)

    def test_untyped_storage_hpu(self):
        npu_storage = torch.empty(
            8,
            dtype=torch.float32,
            device="npu",
        ).untyped_storage()

        self.assertEqual(
            npu_storage.device.type,
            "npu",
        )

        # Scenario 5: Call hpu() with default arguments on the torch-npu platform
        with self.assertRaisesRegex(
            AssertionError,
            r"HPU device module is not loaded",
        ):
            npu_storage.hpu()

        # Scenario 6: Specify device and non_blocking when calling hpu() on the torch-npu platform
        with self.assertRaisesRegex(
            AssertionError,
            r"HPU device module is not loaded",
        ):
            npu_storage.hpu(
                device=1,
                non_blocking=True,
            )

    def test_untyped_storage_nbytes_for_tensor_dtypes(self):
        test_cases = [
            (torch.float16, 6),
            (torch.float32, 6),
            (torch.int32, 6),
            (torch.int64, 6),
        ]

        # Scenarios 7 to 10: CPU/NPU storage byte counts for different dtypes
        for dtype, numel in test_cases:
            with self.subTest(dtype=dtype):
                cpu_tensor = torch.empty(
                    numel,
                    dtype=dtype,
                    device="cpu",
                )
                npu_tensor = torch.empty(
                    numel,
                    dtype=dtype,
                    device="npu",
                )

                cpu_storage = cpu_tensor.untyped_storage()
                npu_storage = npu_tensor.untyped_storage()

                expected_nbytes = (
                    numel * cpu_tensor.element_size()
                )

                self.assertEqual(
                    cpu_storage.nbytes(),
                    expected_nbytes,
                )
                self.assertEqual(
                    npu_storage.nbytes(),
                    expected_nbytes,
                )
                self.assertEqual(
                    cpu_storage.nbytes(),
                    npu_storage.nbytes(),
                )

                # The length of UntypedStorage is also measured in bytes
                self.assertEqual(
                    len(cpu_storage),
                    expected_nbytes,
                )
                self.assertEqual(
                    len(npu_storage),
                    expected_nbytes,
                )

    def test_untyped_storage_nbytes_for_direct_storage(self):
        # Scenario 11: Directly construct CPU/NPU storage with a specified byte count
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                storage = torch.UntypedStorage(
                    37,
                    device=device,
                )

                self.assertEqual(
                    storage.device.type,
                    device,
                )
                self.assertEqual(storage.nbytes(), 37)
                self.assertEqual(storage.size(), 37)
                self.assertEqual(len(storage), 37)

    def test_untyped_storage_nbytes_for_empty_storage(self):
        # Scenario 12: Empty NPU storage has zero bytes
        empty_npu_storage = torch.UntypedStorage(
            0,
            device="npu",
        )
        self.assertEqual(
            empty_npu_storage.nbytes(),
            0,
        )
        self.assertEqual(
            empty_npu_storage.size(),
            0,
        )
        self.assertEqual(
            len(empty_npu_storage),
            0,
        )

    def test_untyped_storage_nbytes_for_tensor_view(self):
        # Scenario 13: A tensor view reports the byte count of the entire underlying storage
        base_tensor = torch.empty(
            (4, 4),
            dtype=torch.float32,
            device="npu",
        )
        view_tensor = base_tensor[:, ::2]

        base_storage = base_tensor.untyped_storage()
        view_storage = view_tensor.untyped_storage()

        base_expected_nbytes = (
            base_tensor.numel()
            * base_tensor.element_size()
        )
        view_logical_nbytes = (
            view_tensor.numel()
            * view_tensor.element_size()
        )

        self.assertEqual(
            base_storage.nbytes(),
            base_expected_nbytes,
        )
        self.assertEqual(
            view_storage.data_ptr(),
            base_storage.data_ptr(),
        )
        self.assertEqual(
            view_storage.nbytes(),
            base_storage.nbytes(),
        )
        self.assertGreater(
            view_storage.nbytes(),
            view_logical_nbytes,
        )

    def test_untyped_storage_new_for_cpu_and_npu(self):
        # Verify the basic properties of storage returned by new() on CPU and NPU
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                source_storage = torch.empty(
                    8,
                    dtype=torch.float32,
                    device=device,
                ).untyped_storage()

                source_nbytes = source_storage.nbytes()
                new_storage = source_storage.new()

                self.assertIsInstance(
                    new_storage,
                    torch.UntypedStorage,
                )
                self.assertIsNot(
                    new_storage,
                    source_storage,
                )
                self.assertEqual(
                    new_storage.device.type,
                    source_storage.device.type,
                )
                self.assertEqual(
                    new_storage.nbytes(),
                    0,
                )
                self.assertEqual(
                    len(new_storage),
                    0,
                )

                # Calling new() must not modify the original storage
                self.assertEqual(
                    source_storage.nbytes(),
                    source_nbytes,
                )

    def test_untyped_storage_new_invalid_argument(self):
        # new() does not accept arguments such as size
        source_storage = torch.empty(
            8,
            dtype=torch.float32,
            device="npu",
        ).untyped_storage()

        with self.assertRaises(TypeError):
            source_storage.new(1)


if __name__ == "__main__":
    run_tests()
