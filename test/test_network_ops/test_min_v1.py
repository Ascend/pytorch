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


class TestMinV1(TestCase):
    def cpu_op_exec(self, data, dim):
        outputs, indices = torch.min(data, dim)
        return outputs.detach()

    def npu_op_exec(self, data, dim):
        data = data.to("npu")
        outputs, indices = torch_npu.npu_min(data, dim)
        return outputs.detach().cpu()

    def test_min_v1_fp32(self, device="npu"):
        data = torch.randn(2, 2, 2, 2, dtype=torch.float32)
        npu_data = data.clone()
        cpu_out = self.cpu_op_exec(data, 2)
        npu_out = self.npu_op_exec(npu_data, 2)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
