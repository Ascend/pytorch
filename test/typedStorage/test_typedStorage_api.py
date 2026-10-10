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
Add validation cases for torch.TypedStorage APIs on NPU.

This file validates torch.TypedStorage.is_shared,
torch.TypedStorage.is_pinned, and torch.TypedStorage.is_hpu
on CPU and NPU.
"""

import warnings

import torch

from torch_npu.testing.testcase import TestCase, run_tests


# TypedStorage is marked as deprecated in PyTorch 2.7, 2.11, and 2.12,
# but this test aims to validate TypedStorage APIs, so the deprecation warning is filtered.
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated.*",
    category=UserWarning,
)


class TestTypedStorageAPIs(TestCase):

    @staticmethod
    def _get_typed_storage(tensor):
        return tensor.storage()

    def test_typed_storage_is_shared_for_cpu(self):
        cpu_tensor = torch.arange(
            8,
            dtype=torch.float32,
            device="cpu",
        )
        expected_tensor = cpu_tensor.clone()
        cpu_storage = self._get_typed_storage(cpu_tensor)

        self.assertIsInstance(
            cpu_storage,
            torch.TypedStorage,
        )
        self.assertEqual(
            cpu_storage.device.type,
            "cpu",
        )

        # Scenario 1: A normal CPU TypedStorage is not in shared memory
        self.assertFalse(cpu_storage.is_shared())
        self.assertEqual(
            cpu_storage.is_shared(),
            cpu_tensor.untyped_storage().is_shared(),
        )

        # Scenario 2: Move CPU TypedStorage to shared memory
        returned_storage = cpu_storage.share_memory_()

        self.assertIs(
            returned_storage,
            cpu_storage,
        )
        self.assertTrue(cpu_storage.is_shared())
        self.assertTrue(cpu_tensor.is_shared())
        self.assertTrue(
            cpu_tensor.untyped_storage().is_shared()
        )

        self.assertEqual(
            cpu_tensor,
            expected_tensor,
        )

    def test_typed_storage_is_shared_for_tensor_view(self):
        base_tensor = torch.arange(
            16,
            dtype=torch.float32,
            device="cpu",
        )
        view_tensor = base_tensor[::2]

        base_storage = self._get_typed_storage(
            base_tensor
        )
        view_storage = self._get_typed_storage(
            view_tensor
        )

        # Scenario 3: Base Tensor and View share the same underlying Storage
        self.assertEqual(
            base_storage.data_ptr(),
            view_storage.data_ptr(),
        )
        self.assertFalse(base_storage.is_shared())
        self.assertFalse(view_storage.is_shared())

        base_storage.share_memory_()

        self.assertTrue(base_storage.is_shared())
        self.assertTrue(view_storage.is_shared())
        self.assertTrue(base_tensor.is_shared())
        self.assertTrue(view_tensor.is_shared())
        self.assertEqual(
            base_storage.data_ptr(),
            view_storage.data_ptr(),
        )

    def test_typed_storage_is_shared_for_npu(self):
        npu_tensor = torch.empty(
            16,
            dtype=torch.float32,
            device="npu",
        )
        npu_storage = self._get_typed_storage(
            npu_tensor
        )

        self.assertEqual(
            npu_storage.device.type,
            "npu",
        )

        is_shared_before = npu_storage.is_shared()
        data_ptr_before = npu_storage.data_ptr()
        nbytes_before = npu_storage.nbytes()

        self.assertIsInstance(
            is_shared_before,
            bool,
        )
        self.assertEqual(
            is_shared_before,
            npu_tensor.untyped_storage().is_shared(),
        )

        returned_storage = npu_storage.share_memory_()
        is_shared_after = npu_storage.is_shared()

        self.assertIs(
            returned_storage,
            npu_storage,
        )
        self.assertEqual(
            is_shared_after,
            is_shared_before,
        )
        self.assertEqual(
            is_shared_after,
            npu_tensor.untyped_storage().is_shared(),
        )
        self.assertEqual(
            npu_storage.data_ptr(),
            data_ptr_before,
        )
        self.assertEqual(
            npu_storage.nbytes(),
            nbytes_before,
        )

    def test_typed_storage_is_pinned_for_normal_storage(self):
        test_dtypes = [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.int64,
        ]

        for device in ["cpu", "npu"]:
            for dtype in test_dtypes:
                with self.subTest(
                    device=device,
                    dtype=dtype,
                ):
                    tensor = torch.empty(
                        16,
                        dtype=dtype,
                        device=device,
                    )
                    storage = self._get_typed_storage(
                        tensor
                    )

                    result_by_string = storage.is_pinned(
                        device="npu"
                    )
                    result_by_device = storage.is_pinned(
                        device=torch.device("npu")
                    )
                    untyped_result = (
                        tensor.untyped_storage().is_pinned(
                            device="npu"
                        )
                    )

                    self.assertIsInstance(
                        result_by_string,
                        bool,
                    )
                    self.assertFalse(result_by_string)
                    self.assertFalse(result_by_device)
                    self.assertEqual(
                        result_by_string,
                        result_by_device,
                    )
                    self.assertEqual(
                        result_by_string,
                        untyped_result,
                    )

    def test_typed_storage_is_pinned_for_pinned_cpu(self):
        test_dtypes = [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.int64,
        ]

        # Scenario 6: Convert CPU TypedStorage to NPU pinned memory
        for dtype in test_dtypes:
            with self.subTest(dtype=dtype):
                source_tensor = torch.arange(
                    1,
                    9,
                    dtype=dtype,
                    device="cpu",
                )
                source_storage = (
                    self._get_typed_storage(
                        source_tensor
                    )
                )
                source_values = (
                    source_storage.tolist()
                )

                self.assertFalse(
                    source_storage.is_pinned(
                        device="npu"
                    )
                )

                pinned_storage = (
                    source_storage.pin_memory(
                        device="npu"
                    )
                )

                self.assertIsInstance(
                    pinned_storage,
                    torch.TypedStorage,
                )
                self.assertIsNot(
                    pinned_storage,
                    source_storage,
                )
                self.assertEqual(
                    pinned_storage.device.type,
                    "cpu",
                )
                self.assertEqual(
                    pinned_storage.dtype,
                    source_storage.dtype,
                )
                self.assertEqual(
                    pinned_storage.size(),
                    source_storage.size(),
                )
                self.assertEqual(
                    pinned_storage.nbytes(),
                    source_storage.nbytes(),
                )
                self.assertEqual(
                    pinned_storage.tolist(),
                    source_values,
                )

                self.assertTrue(
                    pinned_storage.is_pinned(
                        device="npu"
                    )
                )
                self.assertTrue(
                    pinned_storage.is_pinned(
                        device=torch.device("npu")
                    )
                )
                self.assertEqual(
                    pinned_storage.is_pinned(
                        device="npu"
                    ),
                    pinned_storage.untyped().is_pinned(
                        device="npu"
                    ),
                )

                self.assertFalse(
                    source_storage.is_pinned(
                        device="npu"
                    )
                )

    def test_typed_storage_is_hpu_for_cpu_and_npu(self):
        # Scenarios 7 and 8: CPU/NPU TypedStorage does not belong to HPU
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                tensor = torch.empty(
                    16,
                    dtype=torch.float32,
                    device=device,
                )
                storage = self._get_typed_storage(
                    tensor
                )

                self.assertEqual(
                    storage.device.type,
                    device,
                )
                self.assertIsInstance(
                    storage.is_hpu,
                    bool,
                )
                self.assertFalse(storage.is_hpu)
                self.assertEqual(
                    storage.is_hpu,
                    tensor.untyped_storage().is_hpu,
                )


if __name__ == "__main__":
    run_tests()
