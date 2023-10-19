import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGluGrad(TestCase):
    def cpu_op_exec(self, input_data, dim):
        sign = False
        if input_data.dtype == torch.float16:
            input_data = input_data.to(torch.float32)
            sign = True

        input_data.requires_grad = True
        data = torch.nn.functional.glu(input_data, dim=dim)
        data.backward(torch.ones_like(data))
        cpu_output = input_data.grad

        if sign:
            cpu_output = cpu_output.to(torch.float16)

        return cpu_output.to("cpu").numpy()

    def npu_op_exec(self, input_data, dim):
        input_data = input_data.to("npu")
        input_data.requires_grad = True
        data = torch.nn.functional.glu(input_data, dim=dim)
        data.backward(torch.ones_like(data))
        npu_output = input_data.grad

        return npu_output.to("cpu").numpy()

    def test_glugrad_shape_format(self, device="npu"):
        shape_format_32 = [
            [np.float32, -1, (2, 2, 4), 0],
            [np.float32, -1, (4, 6, 10), 1],
            [np.float32, -1, (2, 4, 8), 2],
            [np.float32, -1, (4, 6), -1],
            [np.float32, -1, (2, 2, 4), 2],
            [np.float32, -1, (4, 6, 8, 10), -2],
            [np.float32, -1, (4, 6, 6), 1],
            [np.float32, -1, (6, 20, 10), 1],
        ]

        shape_format_16 = [
            [np.float16, -1, (2, 2, 4), 0],
            [np.float16, -1, (4, 6, 10), 1],
            [np.float16, -1, (2, 4, 8), 2],
            [np.float16, -1, (4, 6), -1],
            [np.float16, -1, (2, 2, 4), 2],
            [np.float16, -1, (4, 6, 8, 10), -2],
            [np.float16, -1, (4, 6, 6), 1],
        ]
        for item in shape_format_32:
            cpu_input, npu_input = create_common_tensor(item, -2.0, 2.0)
            cpu_output = self.cpu_op_exec(cpu_input, item[3])
            npu_output = self.npu_op_exec(npu_input, item[3])
            eps = 0.0002 if item[0].dtype == torch.float32 else 0.002
            self.assertRtolEqual(cpu_output, npu_output, prec=eps)

        for item in shape_format_16:
            cpu_input, npu_input = create_common_tensor(item, -2.0, 2.0)
            cpu_output = self.cpu_op_exec(cpu_input, item[3])
            npu_output = self.npu_op_exec(npu_input, item[3])
            eps = 0.0002 if item[0].dtype == torch.float32 else 0.002
            self.assertRtolEqual(cpu_output, npu_output, prec=eps)


if __name__ == "__main__":
    run_tests()
