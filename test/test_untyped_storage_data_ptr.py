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
Add validation cases for torch.UntypedStorage.data_ptr on NPU.

This file validates data pointer retrieval from NPU storage, including
basic pointer access, different dtypes, and shared storage behavior.
"""

import torch

from torch_npu.testing.testcase import TestCase, run_tests


class TestUntypedStorageDataPtr(TestCase):

    def test_data_ptr_basic(self):
        tensor = torch.arange(8, dtype=torch.float32, device="npu")
        storage = tensor.untyped_storage()

        data_ptr = storage.data_ptr()

        self.assertEqual(storage.device.type, "npu")
        self.assertIsInstance(data_ptr, int)
        self.assertNotEqual(data_ptr, 0)
        self.assertEqual(data_ptr, storage.data_ptr())

    def test_data_ptr_dtypes(self):
        dtypes = [torch.float16, torch.float32, torch.int32]

        for dtype in dtypes:
            tensor = torch.ones(8, dtype=dtype, device="npu")
            data_ptr = tensor.untyped_storage().data_ptr()

            self.assertIsInstance(data_ptr, int)
            self.assertNotEqual(data_ptr, 0)

    def test_data_ptr_shared_storage(self):
        tensor = torch.arange(16, dtype=torch.float32, device="npu")
        tensor_view = tensor[4:12]

        tensor_ptr = tensor.untyped_storage().data_ptr()
        view_ptr = tensor_view.untyped_storage().data_ptr()

        self.assertEqual(tensor_ptr, view_ptr)


if __name__ == "__main__":
    run_tests()
