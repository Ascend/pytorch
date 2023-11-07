import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSortWithoutIndices(TestCase):
    def cpu_dim_op_exec(self, input1, dim, descending):
        output, _ = torch.sort(input1, dim=dim, descending=descending)
        output = output.to(torch.float16).numpy()
        return output

    def npu_dim_op_exec(self, input1, dim, descending):
        output = torch_npu.npu_sort_v2(input1, dim=dim, descending=descending)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_default_op_exec(self, input1):
        output, _ = torch.sort(input1)
        output = output.to(torch.float16).numpy()
        return output

    def npu_default_op_exec(self, input1):
        output = torch_npu.npu_sort_v2(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, descending):
        output, _ = torch.sort(input1, descending=descending)
        output = output.to(torch.float16).numpy()
        return output

    def npu_op_exec(self, input1, descending):
        output = torch_npu.npu_sort_v2(input1, descending=descending)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_sort_v2_dim_shape_format(self):
        shape_format = [
            [[np.float16, 0, (2, 5000)], 0, True],
            [[np.float16, 0, (2, 2, 50000)], 1, False],
            [[np.float16, 0, (2, 289600)], 0, False],
            [[np.float16, 0, (2, 409600)], -1, True],
            [[np.float16, 0, (2, 6, 5)], 1, False],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_dim_op_exec(cpu_input1.to(torch.float), item[1], item[2])
            npu_output = self.npu_dim_op_exec(npu_input1, item[1], item[2])

            self.assertRtolEqual(cpu_output, npu_output)

    def test_sort_v2_shape_format(self):
        shape_format = [
            [[np.float16, 0, (1, 5000)]],
            [[np.float16, 0, (1, 50000)]],
            [[np.float16, 0, (1, 289600)], False],
            [[np.float16, 0, (1, 409600)], True]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) == 1:
                cpu_output = self.cpu_default_op_exec(cpu_input1.to(torch.float))
                npu_output = self.npu_default_op_exec(npu_input1)
            else:
                cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float), item[1])
                npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sort_v2_shape_format_big_range(self):
        shape_format = [
            [[np.float16, 0, (1, 5000)]],
            [[np.float16, 0, (1, 50000)]],
            [[np.float16, 0, (1, 289600)], False],
            [[np.float16, 0, (1, 409600)], True]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -60000, 60000)
            if len(item) == 1:
                cpu_output = self.cpu_default_op_exec(cpu_input1.to(torch.float))
                npu_output = self.npu_default_op_exec(npu_input1)
            else:
                cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float), item[1])
                npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
