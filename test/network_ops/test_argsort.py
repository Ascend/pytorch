import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestArgSort(TestCase):
    def cpu_op_exec(self, input1, dim, descending):
        output = torch.argsort(input1, dim=dim, descending=descending)
        return output.numpy()

    def npu_op_exec(self, input1, dim, descending):
        output = torch.argsort(input1, dim=dim, descending=descending)

        return output.cpu().numpy()

    def cpu_default_op_exec(self, input1):
        output = torch.argsort(input1)
        return output.numpy()

    def npu_default_op_exec(self, input1):
        output = torch.argsort(input1)
        return output.cpu().numpy()

    def test_sort_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (8, 4, 3, 9)], 2, False],
            [[np.float32, 0, (2, 3)]],
            [[np.float32, 0, (1, 7)], 0, True],
            [[np.float32, 0, (1, 5, 6)], 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) > 1:
                cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
                npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            else:
                cpu_output = self.cpu_default_op_exec(cpu_input1)
                npu_output = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sort_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (8, 4, 3, 9)], 2, False],
            [[np.float16, 0, (2, 3)]],
            [[np.float16, 0, (1, 7)], 0, True],
            [[np.float16, 0, (1, 5, 6)], 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) > 1:
                cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float32), item[1], item[2])
                npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            else:
                cpu_output = self.cpu_default_op_exec(cpu_input1.to(torch.float32))
                npu_output = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
