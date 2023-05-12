import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuMish(TestCase):

    def cpu_op_exec(self, input1):
        output = input1 * \
            torch.nn.functional.tanh(torch.nn.functional.softplus(input1))
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch_npu.npu_mish(input1)
        output = output.cpu().numpy()
        return output

    def test_mish(self):
        input1 = torch.randn(5, 5).npu()
        cpu_out = self.cpu_op_exec(input1)
        npu_out = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
