# Copyright 2021 Huawei Technologies Co., Ltd
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
import torch
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import instantiate_device_type_tests, Dtypes, Formats
from torch_npu.testing.util_test import create_dtype_tensor


# For testing TestCase methods and torch_npu.testing functions
class TestTesting(TestCase):
    # Ensure that assertEqual handles cpu arrays properly
    @Dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.bool,
            torch.complex64, torch.complex128)
    @Formats(0, 3, 4, 29)
    def test_assert_equal_cpu(self, device, dtype, npu_format):
        S = 10
        test_sizes = [
            (),
            (0,),
            (S,),
            (S, S),
            (0, S),
            (S, 0)
        ]
        for test_size in test_sizes:
            a_cpu, a_npu = create_dtype_tensor(test_size, dtype, npu_format, device=device)
            msg = f'Device: {device} Size: {test_size} Dtype: {dtype} Npu_format: {npu_format}'
            self.assertEqual(a_cpu, a_npu, message=msg)
        
        
instantiate_device_type_tests(TestTesting, globals(), except_for="cpu")

if __name__ == '__main__':
    run_tests()