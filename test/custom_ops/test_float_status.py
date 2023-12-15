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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestFloatStatus(TestCase):

    def test_float_status(self, device="npu"):
        float_tensor = torch.tensor([40000.0], dtype=torch.float16).npu()
        float_tensor = float_tensor + float_tensor

        input1 = torch.zeros(8).npu()
        float_status = torch_npu.npu_alloc_float_status(input1)
        local_float_status = torch_npu.npu_get_float_status(float_status)

        self.assertTrue(local_float_status.cpu()[0] != 0)
        cleared_float_status = torch_npu.npu_clear_float_status(local_float_status)
        input1 = torch.zeros(8).npu()
        float_status = torch_npu.npu_alloc_float_status(input1)
        local_float_status = torch_npu.npu_get_float_status(float_status)
        self.assertTrue(local_float_status.cpu()[0] == 0)


if __name__ == "__main__":
    run_tests()
