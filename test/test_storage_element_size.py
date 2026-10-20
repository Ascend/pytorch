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

"""Add NPU validation cases for Storage APIs.

PyTorch community lacks complete direct validation for torch.UntypedStorage.element_size.
This file can cover related Storage APIs.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestStorageElementSize(TestCase):

    def test_untyped_storage_element_size(self):
        for dtype in (torch.uint8, torch.int32, torch.float32):
            for size in (0, 2):
                source = torch.empty(size, dtype=dtype, device=device_type).untyped_storage()
                self.assertEqual(source.device.type, device_type)
                self.assertEqual(source.element_size(), 1)
        with self.assertRaises(TypeError):
            source.element_size(1)


if __name__ == "__main__":
    run_tests()
