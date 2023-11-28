import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuLinear(TestCase):
    def cpu_op_exec(self, x, weight, bias):
        output = torch.nn.functional.linear(x, weight, bias)
        output = output.numpy()
        return output

    def npu_op_exec(self, x, weight, bias):
        output = torch_npu.npu_linear(x, weight, bias)
        output = output.cpu().numpy()
        return output

    def test_npu_linear_shape_format_fp32(self):
        shape_format = [
            [[np.float16, -1, (6144, 1024)], [np.float16, -1, (256, 1024)], [np.float16, -1, (256)]],
            [[np.float16, -1, (123, 456)], [np.float16, -1, (789, 456)], [np.float16, -1, (789)]],
        ]

        for item in shape_format:
            cpu_x, npu_x = create_common_tensor(item[0], -2, 2)
            cpu_w, npu_w = create_common_tensor(item[1], -2, 2)
            cpu_b, npu_b = create_common_tensor(item[2], -2, 2)
            cpu_output = self.cpu_op_exec(cpu_x.float(), cpu_w.float(), cpu_b.float())
            npu_output = self.npu_op_exec(npu_x.float(), npu_w.float(), npu_b.float())
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

    def test_npu_linear_shape_format_fp16(self):
        shape_format = [
            [[np.float16, -1, (6144, 1024)], [np.float16, -1, (256, 1024)], [np.float16, -1, (256)]],
            [[np.float16, -1, (123, 456)], [np.float16, -1, (789, 456)], [np.float16, -1, (789)]],
        ]

        for item in shape_format:
            cpu_x, npu_x = create_common_tensor(item[0], -2, 2)
            cpu_w, npu_w = create_common_tensor(item[1], -2, 2)
            cpu_b, npu_b = create_common_tensor(item[2], -2, 2)
            cpu_output = self.cpu_op_exec(cpu_x.float(), cpu_w.float(), cpu_b.float()).astype(np.float16)
            npu_output = self.npu_op_exec(npu_x, npu_w, npu_b)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
