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


class TestBatchNormElemt(TestCase):
    def test_batch_norm_elent(self):
        input1 = torch.tensor([[1.], [2.], [3.]]).npu()
        weight = torch.tensor([1.]).npu()
        bias = torch.tensor([10.]).npu()
        mean = torch.tensor([2.]).npu()
        invstd = torch.tensor([2.]).npu()
        eps = 1e-5
        out = torch.batch_norm_elemt(input1, weight, bias, mean, invstd, eps)
        expect_out = torch.tensor([[8.], [10.], [12.]])
        self.assertRtolEqual(expect_out, out.cpu())

    def test_batch_norm_elent_out(self):
        input1 = torch.tensor([[1.], [2.], [3.]]).npu()
        weight = torch.tensor([1.]).npu()
        bias = torch.tensor([10.]).npu()
        mean = torch.tensor([2.]).npu()
        invstd = torch.tensor([2.]).npu()
        eps = 1e-5
        out = torch.randn((3, 1), dtype=torch.float32).npu()
        torch.batch_norm_elemt(input1, weight, bias, mean, invstd, eps, out=out)
        expect_out = torch.tensor([[8.], [10.], [12.]])
        self.assertRtolEqual(expect_out, out.cpu())


if __name__ == "__main__":
    run_tests()
