import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestFastGelu(TestCase):
    def npu_op_exec(self, input1):
        output = torch_npu.fast_gelu(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_fastgelu(self, device="npu"):
        input1 = torch.tensor([1., 2., 3., 4.]).npu()
        exoutput = torch.tensor([0.8458, 1.9357, 2.9819, 3.9956])
        output = self.npu_op_exec(input1)
        self.assertRtolEqual(exoutput.numpy(), output)


if __name__ == "__main__":
    run_tests()
