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
from torch_npu.testing.common_utils import create_common_tensor

class TestApplyAdam(TestCase):
    def test_apply_adam(self):
        var1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        m1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        v1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        grad1 = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        var2 = var1.to(torch.half)
        m2 = m1.to(torch.half)
        v2 = v1.to(torch.half)
        grad2 = grad1.to(torch.half)
        res1, _, v1_o = torch_npu.npu_apply_adam(1, 1, 0.2, 0.2, 0.2, 0.2, grad1, False, False, out = (var1, m1, v1))
        res2, _, v2_o = torch_npu.npu_apply_adam(1, 1, 0.2, 0.2, 0.2, 0.2, grad2, False, False, out = (var2, m2, v2))
        expect_vo = torch.tensor([[[[1.7452, 0.1779],
                                    [1.6296, 3.0590]],
                                   [[1.7282, 0.0648],
                                    [0.6864, 0.4539]]],
                                   [[[1.5883, 2.6426],
                                    [0.3080, 0.1884]],
                                   [[0.3690, 1.9991],
                                    [3.0633, 0.4669]]]], dtype = torch.float32)
        self.assertRtolEqual(expect_vo, v1_o.cpu())
        self.assertRtolEqual(expect_vo.to(torch.half), v2_o.cpu())

if __name__ == "__main__":
    run_tests()
