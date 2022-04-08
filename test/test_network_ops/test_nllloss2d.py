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
import copy
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNllloss2d(TestCase):
    def cpu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def npu_op_exec(self, data, target, reduction):
        loss = torch.nn.NLLLoss2d(reduction=reduction)
        output = loss(data, target)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def test_nll_loss2d_mean(self, device="npu"):
        m = torch.nn.LogSoftmax(dim=1)
        dim_n, dim_c = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, dim_c, (3, 3))
        data = m(conv(torch.randn(dim_n, 16, 10, 10)))
        target = torch.empty(dim_n, 8, 8, dtype=torch.long).random_(0, dim_c)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)

        cpu_output = self.cpu_op_exec(data, target, "mean")
        npu_output = self.npu_op_exec(data_npu, target_npu, "mean")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_none(self, device="npu"):
        exp = torch.nn.LogSoftmax(dim=1)
        dim_n, dim_c = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, dim_c, (3, 3))
        data = exp(conv(torch.randn(dim_n, 16, 10, 10)))
        target = torch.empty(dim_n, 8, 8, dtype=torch.long).random_(0, dim_c)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)

        cpu_output = self.cpu_op_exec(data, target, "none")
        npu_output = self.npu_op_exec(data_npu, target_npu, "none")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_sum(self, device="npu"):
        exp = torch.nn.LogSoftmax(dim=1)
        dim_n, dim_c = 5, 4
        loss = torch.nn.NLLLoss()
        conv = torch.nn.Conv2d(16, dim_c, (3, 3))
        data = exp(conv(torch.randn(dim_n, 16, 10, 10)))
        target = torch.empty(dim_n, 8, 8, dtype=torch.long).random_(0, dim_c)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)

        cpu_output = self.cpu_op_exec(data, target, "sum")
        npu_output = self.npu_op_exec(data_npu, target_npu, "sum")

        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_nll_loss2d_case_in_ssdresnet34(self, device="npu"):
        cpu_plabel = torch.rand(32, 81, 8732).uniform_(-2.3, 2)
        cpu_glabel = torch.rand(32, 8732).random_(0, 79).long()
        npu_plabel = cpu_plabel.npu()
        npu_glabel = cpu_glabel.npu()

        cpu_con_loss = torch.nn.CrossEntropyLoss(reduce=False)
        npu_con_loss = copy.deepcopy(cpu_con_loss).npu()

        cpu_con = cpu_con_loss(cpu_plabel, cpu_glabel)
        npu_con = npu_con_loss(npu_plabel, npu_glabel)

        self.assertRtolEqual(cpu_con, npu_con.cpu())


if __name__ == "__main__":
    run_tests()
