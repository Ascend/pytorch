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
        expect_vo = torch.tensor([[[[2.2156, -0.1393],
                                    [0.6441, 0.3087]],
                                   [[0.9008, -0.0295],
                                    [0.0776, 0.0773]]],
                                   [[[0.1105, 1.0725],
                                    [0.8731, 0.4582]],
                                   [[0.1653, 0.3091],
                                    [0.3175, 0.0998]]]], dtype = torch.float32)
        self.assertRtolEqual(expect_vo, v1_o.cpu())
        self.assertRtolEqual(expect_vo.to(torch.half), v2_o.cpu())

if __name__ == "__main__":
    run_tests()
