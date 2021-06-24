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
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests

class TestMaxV1(TestCase):
    def cpu_op_exec(self, data, dim):
        outputs, indices = torch.max(data, dim)
        return outputs.detach()

    def npu_op_exec(self, data, dim):
        data = data.to("npu")
        outputs, indices = torch.npu_max(data, dim)
        return outputs.detach().cpu()

    def test_max_v1_fp32(self, device):
        data = torch.randn(2, 2, 2, 2, dtype = torch.float32)
        npu_data = data.clone()
        cpu_out = self.cpu_op_exec(data, 2)
        npu_out = self.npu_op_exec(npu_data, 2)
        self.assertRtolEqual(cpu_out, npu_out)

instantiate_device_type_tests(TestMaxV1, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
