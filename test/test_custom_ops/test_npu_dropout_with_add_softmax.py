import torch
from torch import _VF
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestDropoutWithAddSoftmax(TestCase):
    def cpu_to_exec(self, x1, x2, alpha, prob, axis):
        add_out = torch.add(x1.float(), x2.float(), alpha=alpha)
        softmax_out = torch.nn.functional.softmax(add_out, dim=axis).half()
        output = _VF.dropout(softmax_out.float(), prob, False).half()
        return softmax_out, output

    def npu_to_exec(self, x1, x2, alpha, prob, dim):
        _, softmax_out, output = torch_npu.npu_dropout_with_add_softmax(x2, x1, alpha, prob, dim)
        return softmax_out, output

    def test_dropout_with_add_softmax_fp32(self):
        x1 = torch.randn(32, 12, 384, 384, dtype=torch.float32).npu()
        x2 = torch.randn(32, 12, 384, 384, dtype=torch.float32).npu()
        alpha = 0.1
        axis = -1
        prob = 0

        _, cpu_out = self.cpu_to_exec(x1, x2, alpha, prob, axis)
        _, npu_out = self.npu_to_exec(x1, x2, alpha, prob, axis)
        cpu_out = cpu_out.float().cpu().numpy()
        npu_out = npu_out.cpu().numpy()

        self.assertRtolEqual(cpu_out, npu_out, prec=0.001)

    def test_dropout_with_add_softmax_fp16(self):
        x1 = torch.randn(32, 12, 384, 384, dtype=torch.float16).npu()
        x2 = torch.randn(32, 12, 384, 384, dtype=torch.float16).npu()
        alpha = 0.12
        axis = -1
        prob = 0

        _, cpu_out = self.cpu_to_exec(x1, x2, alpha, prob, axis)
        _, npu_out = self.npu_to_exec(x1, x2, alpha, prob, axis)
        cpu_out = cpu_out.cpu().numpy()
        npu_out = npu_out.cpu().numpy()

        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
