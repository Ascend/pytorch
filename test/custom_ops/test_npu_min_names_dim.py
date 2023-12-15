# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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


class TestMinNamesDim(TestCase):
    def cpu_op_exec(self, data, dim, keepdim=False):
        outputs, indices = torch.min(data, dim, keepdim)
        return outputs.cpu().numpy(), indices.int().cpu().numpy()

    def npu_op_exec(self, data, dim, keepdim=False):
        outputs, indices = torch_npu.npu_min(data, dim, keepdim)
        return outputs.cpu().numpy(), indices.cpu().numpy()

    def test_min_names_dim_without_keepdim(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float32,
                           names=('A', 'B', 'C', 'D')).npu()
        cpu_value, cpu_indices = self.cpu_op_exec(data, 1)
        npu_value, npu_indices = self.npu_op_exec(data, 'B')
        self.assertRtolEqual(cpu_value, npu_value)
        self.assertRtolEqual(cpu_indices, npu_indices)

    def test_min_names_dim_with_keepdim(self):
        data = torch.randn(2, 3, 4, 5, dtype=torch.float32,
                           names=('A', 'B', 'C', 'D')).npu()
        cpu_value, cpu_indices = self.cpu_op_exec(data, 3, True)
        npu_value, npu_indices = self.npu_op_exec(data, 'D', True)
        self.assertRtolEqual(cpu_value, npu_value)
        self.assertRtolEqual(cpu_indices, npu_indices)


if __name__ == "__main__":
    run_tests()
