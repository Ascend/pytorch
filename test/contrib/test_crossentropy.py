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
from torch_npu.contrib.module import LabelSmoothingCrossEntropy


class TestCrossentropy(TestCase):
    @unittest.skip("skip test_npu_crossentropy_1 now")
    def test_npu_crossentropy_1(self):
        x = torch.randn(2, 10)
        y = torch.randint(0, 10, size=(2,))

        x = x.npu()
        y = y.npu()
        x.requires_grad = True
        m = LabelSmoothingCrossEntropy(10)
        npu_output = m(x, y)
        npu_output.backward()
        expedt_cpu_xgrad = torch.tensor([[0.0465, 0.0317, 0.0612, 0.0215, 0.0695,
                                          0.0849, 0.0354, 0.0255, -0.4017, 0.0255],
                                         [0.0133, 0.0225, 0.0104, 0.0787, 0.0202,
                                          0.1322, -0.4969, 0.1719, 0.0331, 0.0145]], dtype=torch.float32)
        self.assertTrue(3.3496, npu_output.detach().cpu())
        self.assertRtolEqual(expedt_cpu_xgrad, x.grad.cpu())
     
    @unittest.skip("skip test_npu_crossentropy_2 now")
    def test_npu_crossentropy_2(self):
        x = torch.randn(2, 10)
        y = torch.randint(0, 10, size=(2,))

        x = x.npu()
        y = y.npu()
        x.requires_grad = True
        m = LabelSmoothingCrossEntropy(10, 0.1)
        npu_output = m(x, y)
        npu_output.backward()
        expedt_cpu_xgrad = torch.tensor([[0.0410, 0.0261, 0.0557, 0.0160, 0.0639,
                                          0.0793, 0.0298, 0.0200, -0.3517, 0.0199],
                                        [0.0077, 0.0170, 0.0049, 0.0732, 0.0146,
                                         0.1267, -0.4469, 0.1663, 0.0275, 0.0090]], dtype=torch.float32)
        self.assertTrue(3.2760, npu_output.cpu())
        self.assertRtolEqual(expedt_cpu_xgrad, x.grad.cpu())


if __name__ == "__main__":
    run_tests()
