import copy
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


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

    def test_nll_loss2d_mean(self):
        m = torch.nn.LogSoftmax(dim=1)
        dim_n, dim_c = 5, 4
        conv = torch.nn.Conv2d(16, dim_c, (3, 3))
        data = m(conv(torch.randn(dim_n, 16, 10, 10)))
        target = torch.empty(dim_n, 8, 8, dtype=torch.long).random_(0, dim_c)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)

        cpu_output = self.cpu_op_exec(data, target, "mean")
        npu_output = self.npu_op_exec(data_npu, target_npu, "mean")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_none(self):
        exp = torch.nn.LogSoftmax(dim=1)
        dim_n, dim_c = 5, 4
        conv = torch.nn.Conv2d(16, dim_c, (3, 3))
        data = exp(conv(torch.randn(dim_n, 16, 10, 10)))
        target = torch.empty(dim_n, 8, 8, dtype=torch.long).random_(0, dim_c)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)

        cpu_output = self.cpu_op_exec(data, target, "none")
        npu_output = self.npu_op_exec(data_npu, target_npu, "none")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_sum(self):
        exp = torch.nn.LogSoftmax(dim=1)
        dim_n, dim_c = 5, 4
        conv = torch.nn.Conv2d(16, dim_c, (3, 3))
        data = exp(conv(torch.randn(dim_n, 16, 10, 10)))
        target = torch.empty(dim_n, 8, 8, dtype=torch.long).random_(0, dim_c)

        data_npu = data.to("npu")
        target_npu = target.to("npu")
        target_npu = target_npu.to(torch.int32)

        cpu_output = self.cpu_op_exec(data, target, "sum")
        npu_output = self.npu_op_exec(data_npu, target_npu, "sum")

        self.assertRtolEqual(cpu_output, npu_output)

    def test_nll_loss2d_case_in_ssdresnet34(self):
        cpu_plabel = torch.rand(32, 81, 8732).uniform_(-2.3, 2)
        cpu_glabel = torch.rand(32, 8732).random_(0, 79).long()
        npu_plabel = cpu_plabel.npu()
        npu_glabel = cpu_glabel.npu()

        cpu_con_loss = torch.nn.CrossEntropyLoss(reduce=False)
        npu_con_loss = copy.deepcopy(cpu_con_loss).npu()

        cpu_output = cpu_con_loss(cpu_plabel, cpu_glabel)
        npu_output = npu_con_loss(npu_plabel, npu_glabel)

        self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_nll_loss2d_dim_4(self):
        cpu_input, npu_input = create_common_tensor((np.float32, 0, (32, 32, 32, 32)), -100, 100)
        cpu_target, npu_target = create_common_tensor((np.int64, 0, (32, 32, 32)), 0, 32)
        weight, _ = create_common_tensor((np.float32, 0, 32), 0, 32)
        ignore_index = -100
        reductions = ["mean", "sum", "none"]
        for reduction in reductions:
            cpu_loss = torch.nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
            npu_loss = copy.deepcopy(cpu_loss).npu()

            cpu_output = cpu_loss(cpu_input, cpu_target)
            npu_output = npu_loss(npu_input, npu_target)

            self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == "__main__":
    run_tests()
