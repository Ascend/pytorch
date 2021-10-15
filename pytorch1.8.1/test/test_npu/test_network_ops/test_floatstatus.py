# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests

class TestFloatStatus(TestCase):
    def test_float_status(self, device):
        input1 = torch.randn([8]).npu().fill_(2)
        float_status = torch.npu_alloc_float_status(input1)
        temp = float_status.to("cpu")
        local_float_status = torch.npu_get_float_status(float_status)
        exoutput = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0.])
        if(float_status.cpu()[0] != 0):
            cleared_float_status = torch.npu_clear_float_status(local_float_status)
            print("test_float_status overflow!!!")
            self.assertRtolEqual(exoutput.numpy(), cleared_float_status.cpu().numpy())
        else:
            print("test_float_status success~~~")
            self.assertRtolEqual(exoutput.numpy(), temp.numpy())

instantiate_device_type_tests(TestFloatStatus, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
