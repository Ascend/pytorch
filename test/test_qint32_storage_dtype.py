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
Add focused validation cases for torch.QInt32Storage.dtype:
1. PyTorch community covers the property through test_storage_error while
   iterating torch._storage_classes. The registry mixes CPU, CUDA, and NPU
   storage classes in the downstream NPU environment.
2. This file directly validates the class and instance dtype properties.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestQInt32StorageDtype(TestCase):
    def test_class_dtype(self):
        self.assertIs(torch.QInt32Storage.dtype, torch.qint32)

    def test_instance_dtype(self):
        storages = (
            torch.QInt32Storage(),
            torch.QInt32Storage(4),
            torch.QInt32Storage([0, 1, 2]),
        )
        for storage in storages:
            self.assertIs(storage.dtype, torch.qint32)
            self.assertIs(storage.dtype, torch.QInt32Storage.dtype)


if __name__ == "__main__":
    run_tests()
