# Copyright (c) 2022, Huawei Technologies.All rights reserved.
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
from torch_npu.testing.common_utils import check_operators_in_prof


class TestViewCopy(TestCase):

    def test_view_copy_of_slice(self):
        cpu_x = torch.rand(2, 3)
        cpu_other = torch.rand(2, 1)
        npu_x = cpu_x.npu()
        npu_other = cpu_other.npu()
        cpu_slice = cpu_x[:, 1:2]
        cpu_slice.copy_(cpu_other)
        with torch.autograd.profiler.profile(use_npu=True) as prof:
            npu_slice = npu_x[:, 1:2]
            npu_slice.copy_(npu_other)
        self.assertEqual(check_operators_in_prof(['ViewCopy'], prof), True, "Error operators called!")
        self.assertRtolEqual(cpu_slice, npu_slice.cpu())

    def test_view_copy_of_transpose(self):
        cpu_x = torch.rand(2, 3)
        cpu_other = torch.rand(3, 2)
        npu_x = cpu_x.npu()
        npu_other = cpu_other.npu()
        cpu_t = cpu_x.t()
        cpu_t.copy_(cpu_other)
        with torch.autograd.profiler.profile(use_npu=True) as prof:
            npu_t = npu_x.t()
            npu_t.copy_(npu_other)
        self.assertEqual(check_operators_in_prof(['ViewCopy'], prof), True, "Error operators called!")
        self.assertRtolEqual(cpu_t, npu_t.cpu())


if __name__ == "__main__":
    run_tests()
