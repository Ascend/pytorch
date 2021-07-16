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

class TestBroadCastToD(TestCase):
    @dtypes(torch.float, torch.float16, torch.int32, torch.int8, torch.uint8, torch.bool)
    def test_broadcast(self, device, dtype):
        shapes = [
                    [[1], [5]],
                    [[ 1, 2], [3, 2]],
                    [[1, 2, 1], [1, 2, 3]],
                ]
        for item in shapes:
            input1 = torch.randn(item[0]).to(dtype).npu()
            output = input1.npu_broadcast(item[1])
            size1 = np.array(output.size(), dtype=np.int32)
            size2 = np.array(item[1], dtype=np.int32)
            self.assertRtolEqual(size1, size2)


instantiate_device_type_tests(TestBroadCastToD, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()