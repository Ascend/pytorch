# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestRandn(TestCase):
    @dtypes(torch.float64, torch.float, torch.float16)
    def test_randn(self, device, dtype):
        torch.manual_seed(123456)
        res1 = torch.randn((12, 345), dtype=dtype, device='npu')
        res2 = torch.tensor([], dtype=dtype).npu()
        torch.manual_seed(123456)
        torch.randn((12, 345), device='npu', out=res2)
        self.assertRtolEqual(res1.cpu(), res2.cpu())


instantiate_device_type_tests(TestRandn, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()