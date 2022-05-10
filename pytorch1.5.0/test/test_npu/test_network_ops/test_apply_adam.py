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
        bt1p = 0.9
        bt2p = 0.9
        lr = 0.2
        bt1 = 0.2
        bt2 = 0.2
        ep = 0.2
        ul = False
        un = False
        var_o, m_o, v_o = torch.npu_apply_adam(var, m, v, bt1p, bt2p, lr, bt1, bt2, ep, grad, ul, un)
        expect_varo = torch.tensor([[[[-0.7631, -0.4471],
                                      [ 0.8030, -0.2334]],
                                     [[ 0.6573,  0.1233],
                                      [ 0.6493, -0.9261]]],
                                    [[[-0.7625,  0.2999],
                                      [-1.3133, -1.0723]],
                                     [[-0.3252, -0.7887],
                                      [ 0.9137, -1.5635]]]])
        expect_mo = torch.tensor([[[[ 1.4386, -0.0559],
                                    [-1.3626, -1.9319]],
                                   [[-1.1560,  0.4082],
                                    [-1.0249,  0.4093]]],
                                  [[[ 1.0244,  1.1617],
                                    [ 0.5887,  0.6287]],
                                   [[-0.2391, -1.1880],
                                    [-1.6345,  0.8329]]]])
        expect_vo = torch.tensor([[[[1.7452, 0.1779],
                                    [1.6296, 3.0590]],
                                   [[1.7282, 0.0648],
                                    [0.6864, 0.4539]]],
                                   [[[1.5883, 2.6426],
                                    [0.3080, 0.1884]],
                                   [[0.3690, 1.9991],
                                    [3.0633, 0.4669]]]], dtype = torch.float32)
        self.assertRtolEqual(expect_varo, var_o.cpu())
        self.assertRtolEqual(expect_mo, m_o.cpu())
        self.assertRtolEqual(expect_vo, v_o.cpu())

    def test_apply_adam_out_fp32(self, device):
        var = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        m = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        v = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        grad = torch.randn(2, 2, 2, 2, dtype=torch.float32).to("npu")
        bt1p = 0.9
        bt2p = 0.9
        lr = 0.2
        bt1 = 0.2
        bt2 = 0.2
        ep = 0.2
        ul = False
        un = False
        var_o, m_o, v_o = torch.npu_apply_adam(bt1p, bt2p, lr, bt1, bt2, ep, grad, ul, un, out = (var, m, v))
        expect_varo = torch.tensor([[[[-0.7631, -0.4471],
                                      [ 0.8030, -0.2334]],
                                     [[ 0.6573,  0.1233],
                                      [ 0.6493, -0.9261]]],
                                    [[[-0.7625,  0.2999],
                                      [-1.3133, -1.0723]],
                                     [[-0.3252, -0.7887],
                                      [ 0.9137, -1.5635]]]])
        expect_mo = torch.tensor([[[[ 1.4386, -0.0559],
                                    [-1.3626, -1.9319]],
                                   [[-1.1560,  0.4082],
                                    [-1.0249,  0.4093]]],
                                  [[[ 1.0244,  1.1617],
                                    [ 0.5887,  0.6287]],
                                   [[-0.2391, -1.1880],
                                    [-1.6345,  0.8329]]]])
        expect_vo = torch.tensor([[[[1.7452, 0.1779],
                                    [1.6296, 3.0590]],
                                   [[1.7282, 0.0648],
                                    [0.6864, 0.4539]]],
                                   [[[1.5883, 2.6426],
                                    [0.3080, 0.1884]],
                                   [[0.3690, 1.9991],
                                    [3.0633, 0.4669]]]], dtype = torch.float32)
        self.assertRtolEqual(expect_varo, var_o.cpu())
        self.assertRtolEqual(expect_mo, m_o.cpu())
        self.assertRtolEqual(expect_vo, v_o.cpu())

instantiate_device_type_tests(TestApplyAdam, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
