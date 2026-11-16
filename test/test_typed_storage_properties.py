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
Add validation cases for torch.TypedStorage properties on NPU:
1. PyTorch community tests do not sufficiently check is_cuda return values.
2. This file validates torch.TypedStorage.is_cuda and can be extended.
CPU storage provides a reference for device transitions.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTypedStorageProperties(TestCase):

    def check_storage(self, storage, expected_device):
        self.assertIsInstance(storage, torch.TypedStorage)
        self.assertEqual(storage.device.type, expected_device)
        self.assertIs(storage.is_cuda, False)

    def test_is_cuda_from_tensor(self):
        for dtype in (torch.float32, torch.int64, torch.bool):
            for size in (0, 4):
                with self.subTest(dtype=dtype, size=size):
                    tensor = torch.empty(size, dtype=dtype, device=device_type)
                    self.check_storage(tensor.storage(), "npu")

    def test_is_cuda_constructor(self):
        storage = torch.TypedStorage(4, dtype=torch.float32, device=device_type)
        self.check_storage(storage, "npu")

    def test_is_cuda_wrapped_storage(self):
        for size in (0, 4):
            with self.subTest(size=size):
                tensor = torch.empty(size, device=device_type)
                storage = torch.TypedStorage(
                    wrap_storage=tensor.untyped_storage(), dtype=tensor.dtype
                )
                self.check_storage(storage, "npu")

    def test_is_cuda_device_transfer(self):
        tensor = torch.zeros(4, device=device_type)
        self.check_storage(tensor.storage(), "npu")
        cpu_tensor = tensor.cpu()
        self.check_storage(cpu_tensor.storage(), "cpu")
        self.check_storage(cpu_tensor.to(device_type).storage(), "npu")


if __name__ == "__main__":
    run_tests()
