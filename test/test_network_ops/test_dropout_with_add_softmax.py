import torch
import torch.nn.functional as F
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDropOutWithAddSoftMax(TestCase):

    def cpu_op_exec(self, x1, x2, alpha, axis):
        dropout = torch.nn.Dropout(p=0)
        add_out = torch.add(x1.float(), x2.float(), alpha=alpha)
        softmax_out = F.softmax(add_out, dim=axis).half()
        output = dropout(softmax_out.float()).half()
        return softmax_out.detach().numpy(), output.detach().numpy()

    def npu_op_exec(self, x1, x2, alpha, prod, dim):
        _, softmax_out, output = torch_npu.npu_dropout_with_add_softmax(x2, x1, alpha, prod, dim)
        return softmax_out.cpu().detach().numpy(), output.cpu().detach().numpy()

    def test_dropout_shape_format(self):
        dtypes = [torch.half, torch.float]
        for dtype in dtypes:
            cpu_input1 = torch.rand(96, 12, 384, 384).to(dtype)
            cpu_input2 = torch.rand(96, 12, 384, 384).to(dtype)
            npu_input1 = cpu_input1.npu()
            npu_input2 = cpu_input2.npu()
            alpha = 0.125
            axis = -1
            prod_npu = 0

            _, cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, alpha, axis)
            _, npu_output = self.npu_op_exec(npu_input1, npu_input2, alpha, prod_npu, axis)
            if dtype == torch.float:
                cpu_output = cpu_output.astype(np.float32)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
