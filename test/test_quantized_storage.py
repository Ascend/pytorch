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
Add validation cases for torch.QUInt8Storage on NPU:
1. PyTorch community test_storage_error iterates torch._storage_classes and mixes
   CPU, CUDA and NPU storage classes, which makes it unsuitable for focused
   QUInt8Storage validation on NPU.
2. This file validates torch.QUInt8Storage construction (empty, size, data,
   wrap_storage), dtype, element_size, and error handling.
3. Storage construction is device-independent, so tensors are created on CPU
   and only used as invalid constructor arguments.
"""
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestQUInt8Storage(TestCase):
    def test_construct_empty(self):
        storage = torch.QUInt8Storage()
        self.assertEqual(storage.size(), 0)
        self.assertEqual(storage.dtype, torch.QUInt8Storage.dtype)
        self.assertEqual(storage.element_size(), 1)

    def test_construct_with_size(self):
        for size in (0, 1, 16):
            storage = torch.QUInt8Storage(size)
            self.assertEqual(storage.size(), size)
            self.assertEqual(storage.dtype, torch.QUInt8Storage.dtype)
            self.assertEqual(storage.nbytes(), size)

    def test_construct_with_data(self):
        data = [0, 1, 2, 255]
        storage = torch.QUInt8Storage(data)
        self.assertEqual(storage.size(), len(data))
        self.assertEqual(storage.tolist(), data)

    def test_construct_with_wrap_storage(self):
        storage = torch.QUInt8Storage(4)
        wrapped = torch.QUInt8Storage(wrap_storage=storage.untyped())
        self.assertEqual(wrapped.size(), storage.size())
        self.assertEqual(wrapped.dtype, storage.dtype)
        self.assertEqual(wrapped.tolist(), storage.tolist())

    def test_constructor_errors(self):
        with self.assertRaises(RuntimeError):
            torch.QUInt8Storage(device="cpu")
        with self.assertRaises(RuntimeError):
            torch.QUInt8Storage(dtype=torch.float)
        with self.assertRaises(TypeError):
            torch.QUInt8Storage(invalid_argument=torch.float)
        with self.assertRaises(RuntimeError):
            torch.QUInt8Storage(0, 0)
        with self.assertRaises(TypeError):
            torch.QUInt8Storage("string")
        with self.assertRaises(TypeError):
            torch.QUInt8Storage(torch.tensor([]))

    def test_wrap_storage_errors(self):
        storage = torch.QUInt8Storage()
        with self.assertRaises(RuntimeError):
            torch.QUInt8Storage(0, wrap_storage=storage.untyped())
        with self.assertRaises(TypeError):
            torch.QUInt8Storage(wrap_storage=storage)


if __name__ == "__main__":
    run_tests()
