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
Add validation cases for torch.UntypedStorage.copy_ on NPU:
1. PyTorch community lacks direct API validations for UntypedStorage.copy_,
   so this file is added.
2. This file validates the copy_ behavior on CPU/NPU (same-device, cross-device,
   non_blocking, empty storage, size mismatch and the TypedStorage forwarding
   path).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestUntypedStorageCopy(TestCase):

    def test_copy_same_device_cpu(self):
        src = torch.UntypedStorage([1, 2, 3, 4])
        dst = torch.UntypedStorage(4)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.device.type, "cpu")
        self.assertEqual(list(dst), [1, 2, 3, 4])

    def test_copy_same_device_npu(self):
        src = torch.UntypedStorage([1, 2, 3, 4], device=device_type)
        dst = torch.UntypedStorage(4, device=device_type)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.device.type, device_type)
        self.assertEqual(list(dst), [1, 2, 3, 4])

    def test_copy_cpu_to_npu(self):
        src = torch.UntypedStorage([7, 8, 9])
        dst = torch.UntypedStorage(3, device=device_type)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.device.type, device_type)
        self.assertEqual(list(dst), [7, 8, 9])

    def test_copy_npu_to_cpu(self):
        src = torch.UntypedStorage([7, 8, 9], device=device_type)
        dst = torch.UntypedStorage(3)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.device.type, "cpu")
        self.assertEqual(list(dst), [7, 8, 9])

    def test_copy_non_blocking(self):
        src = torch.UntypedStorage([1, 2, 3], device=device_type)
        dst = torch.UntypedStorage(3, device=device_type)
        ret = dst.copy_(src, non_blocking=True)
        self.assertIs(ret, dst)
        self.assertEqual(list(dst), [1, 2, 3])

    def test_copy_empty(self):
        src = torch.UntypedStorage([])
        dst = torch.UntypedStorage(0)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(dst.size(), 0)

    def test_copy_size_mismatch(self):
        src = torch.UntypedStorage(4)
        dst = torch.UntypedStorage(3)
        with self.assertRaisesRegex(RuntimeError, "size does not match"):
            dst.copy_(src)

    def test_copy_typed_storage_forwarding(self):
        src = torch.TypedStorage([1, 2, 3], dtype=torch.float32)
        dst = torch.TypedStorage(3, dtype=torch.float32)
        ret = dst.copy_(src)
        self.assertIs(ret, dst)
        self.assertEqual(list(dst), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    run_tests()
