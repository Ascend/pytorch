import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaxV1(TestCase):
    def cpu_op_exec(self, data, dim):
        outputs, indices = torch.max(data, dim)
        return outputs.detach()

    def npu_op_exec(self, data, dim):
        data = data.to("npu")
        outputs, indices = torch_npu.npu_max(data, dim)
        return outputs.detach().cpu()

    def test_max_v1(self):
        shape_format = [
            [np.float32, -1, (10,)],
            [np.float32, 3, (4, 4, 4)],
            [np.float32, 2, (64, 63)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, 0)
            npu_output = self.npu_op_exec(npu_input, 0)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
