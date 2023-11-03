import torch
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLayerNormEval(TestCase):
    def cpu_op_exec(self, input1):
        m = nn.LayerNorm(input1.size()[1:]).eval()
        output = m(input1)
        return output

    def npu_op_exec(self, input1):
        m = nn.LayerNorm(input1.size()[1:]).npu().eval()
        output = m(input1)
        output = output.to("cpu")
        return output

    def test_layernorm_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 0, (64, 10)],
            [np.float32, 0, (256, 2048, 7, 7)],
            [np.float32, 0, (32, 1, 3, 3)],
            [np.float32, 0, (10, 128)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())


if __name__ == "__main__":
    run_tests()
