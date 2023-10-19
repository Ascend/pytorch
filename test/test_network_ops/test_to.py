import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTo(TestCase):
    def cpu_op_exec(self, input1, target):
        output = input1.to(target)
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1, target):
        output = input1.to(target)
        output = output.cpu().numpy()
        return output

    def test_to(self, device="npu"):
        shape_format = [
            [np.float32, 0, [3, 3]],
            [np.float16, 0, [4, 3]],
            [np.int32, 0, [3, 5]],
        ]
        targets = [torch.float16, torch.float32, torch.int32, 'cpu', 'npu']
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            for target in targets:
                cpu_output = self.cpu_op_exec(cpu_input1, target)
                npu_output = self.npu_op_exec(npu_input1, target)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
