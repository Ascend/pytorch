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

PyTorch community lacks complete direct validation for torch.TypedStorage.complex_double.
This file can cover related Storage APIs.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestStorageComplexDouble(TestCase):

    def test_typed_storage_complex_double(self):
        for values, dtype in (([1 + 2j, -3 + 0.5j], torch.complex64), ([1.25, -2.5], torch.float32)):
            cpu_source = torch.tensor(values, dtype=dtype).storage()
            npu_source = torch.tensor(values, dtype=dtype, device=device_type).storage()
            expected = cpu_source.complex_double()
            actual = npu_source.complex_double()
            self.assertEqual(actual.device.type, device_type)
            self.assertEqual(actual.dtype, torch.complex128)
            self.assertEqual(actual.size(), len(values))
            self.assertEqual(actual.cpu().tolist(), expected.tolist())
            self.assertEqual(npu_source.dtype, dtype)

        empty = torch.empty(0, dtype=torch.complex64, device=device_type).storage()
        self.assertEqual(empty.complex_double().size(), 0)
        with self.assertRaises(TypeError):
            empty.complex_double(torch.float32)



if __name__ == "__main__":
    run_tests()
