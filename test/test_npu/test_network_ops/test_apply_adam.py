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

class TestApplyAdam(TestCase):
    def test_apply_adam_fp32(self, device):
        var = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        m = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        v = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        grad = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        bt1p = 1
        bt2p = 1
        lr = 0.2
        bt1 = 0.2
        bt2 = 0.2
        ep = 0.2
        ul = False
        un = False
        var_o, m_o, v_o = torch.npu_apply_adam(var, m, v, bt1p, bt2p, lr, bt1, bt2, ep, grad, ul, un)
        expect_vo = torch.tensor([[[[1.7452, 0.1779],
                                    [1.6296, 3.0590]],
                                   [[1.7282, 0.0648],
                                    [0.6864, 0.4539]]],
                                   [[[1.5883, 2.6426],
                                    [0.3080, 0.1884]],
                                   [[0.3690, 1.9991],
                                    [3.0633, 0.4669]]]], dtype = torch.float32)
        self.assertRtolEqual(expect_vo, v_o.cpu())

instantiate_device_type_tests(TestApplyAdam, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()

