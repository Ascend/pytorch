# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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
import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import DCNv2


class TestDeformConv(TestCase):
    @unittest.skip("skip test_npu_deform_conv_1 now")
    def test_npu_deform_conv_1(self):
        x = torch.randn(2, 2, 3, 3)
        model = DCNv2(2, 2, 3, 2, 1)

        x = x.npu()
        x.requires_grad = True
        model = model.npu()

        output = model(x)
        output.sum().backward()

        expedt_cpu_output = torch.tensor([[[[0.0359, -0.0297],
                                            [0.2135, 0.0879]],

                                           [[0.2250, 0.0972],
                                            [-0.1099, 0.0224]]],


                                          [[[-0.2693, 0.4263],
                                              [-0.2629, -0.2155]],

                                           [[0.0654, -0.4343],
                                            [-0.0067, 0.1704]]]], dtype=torch.float32)

        expedt_cpu_xgrad = torch.tensor([[[[0.1239, -0.1093, 0.1239],
                                           [-0.0060, 0.2868, -0.0060],
                                           [0.1239, -0.1093, 0.1239]],

                                          [[-0.0404, -0.0229, -0.0404],
                                           [0.0341, -0.3033, 0.0341],
                                           [-0.0404, -0.0229, -0.0404]]],


                                         [[[0.1239, -0.1093, 0.1239],
                                           [-0.0060, 0.2868, -0.0060],
                                           [0.1239, -0.1093, 0.1239]],

                                          [[-0.0404, -0.0229, -0.0404],
                                           [0.0341, -0.3033, 0.0341],
                                             [-0.0404, -0.0229, -0.0404]]]], dtype=torch.float32)
        self.assertRtolEqual(expedt_cpu_output, output.detach().cpu())
        self.assertRtolEqual(expedt_cpu_xgrad, x.grad.cpu())
    
    @unittest.skip("skip test_npu_deform_conv_2 now")
    def test_npu_deform_conv_2(self):
        x = torch.randn(2, 2, 5, 5)
        model = DCNv2(2, 2, 3, 2, 1)

        x = x.npu()
        x.requires_grad = True
        model = model.npu()

        output = model(x)
        output.sum().backward()

        expedt_cpu_output = torch.tensor([[[[0.0568, -0.0061, 0.0660],
                                            [0.0097, -0.1420, 0.1683],
                                            [0.1852, -0.3518, -0.0962]],
                                           [[0.1061, -0.1260, -0.0641],
                                            [-0.1111, -0.4993, 0.2399],
                                            [0.2185, 0.1643, 0.0179]]],
                                          [[[-0.4639, -0.0570, -0.1230],
                                            [-0.1023, 0.3752, 0.5703],
                                            [-0.0807, -0.3313, -0.1658]],
                                           [[-0.1327, -0.2725, -0.0910],
                                            [-0.0751, 0.1237, 0.4084],
                                            [0.1052, -0.2177, -0.0489]]]], dtype=torch.float32)

        expedt_cpu_xgrad = torch.tensor([[[[0.1238, 0.0521, 0.1238, 0.0521, 0.1238],
                                        [-0.0504, 0.2393, -0.0504, 0.2393, -0.0504],
                                        [0.1238, 0.0521, 0.1238, 0.0521, 0.1238],
                                        [-0.0504, 0.2393, -0.0504, 0.2393, -0.0504],
                                        [0.1238, 0.0521, 0.1238, 0.0521, 0.1238]],
            [[-0.1459, -0.0728, -0.1459, -0.0728, -0.1459],
             [-0.0267, -0.3579, -0.0267, -0.3579, -0.0267],
             [-0.1459, -0.0728, -0.1459, -0.0728, -0.1459],
             [-0.0267, -0.3579, -0.0267, -0.3579, -0.0267],
             [-0.1459, -0.0728, -0.1459, -0.0728, -0.1459]]],
            [[[0.1238, 0.0521, 0.1238, 0.0521, 0.1238],
              [-0.0504, 0.2393, -0.0504, 0.2393, -0.0504],
              [0.1238, 0.0521, 0.1238, 0.0521, 0.1238],
              [-0.0504, 0.2393, -0.0504, 0.2393, -0.0504],
              [0.1238, 0.0521, 0.1238, 0.0521, 0.1238]],
             [[-0.1459, -0.0728, -0.1459, -0.0728, -0.1459],
             [-0.0267, -0.3579, -0.0267, -0.3579, -0.0267],
             [-0.1459, -0.0728, -0.1459, -0.0728, -0.1459],
             [-0.0267, -0.3579, -0.0267, -0.3579, -0.0267],
             [-0.1459, -0.0728, -0.1459, -0.0728, -0.1459]]]],
            dtype=torch.float32)
        self.assertRtolEqual(expedt_cpu_output, output.detach().cpu())
        self.assertRtolEqual(expedt_cpu_xgrad, x.grad.cpu())


if __name__ == "__main__":
    run_tests()
