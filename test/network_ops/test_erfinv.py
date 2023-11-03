import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestErfinv(TestCase):
    def get_shapeFormat(self):
        shape_format = [
            [[np.float32, -1, (2, 3, 4, 5)], [np.float32, -1, (2, 3, 4, 5)]],
            [[np.float32, -1, (4, 5, 6, 7)], [np.float32, -1, (4, 5, 6, 7)]],
            [[np.float32, -1, (4, 5, 6, 7)], [np.float32, -1, (2, 3, 4, 5)]],
            [[np.float32, -1, (2, 3, 4, 5, 6)], [np.float32, -1, (2, 3, 5, 6, 7)]],
            [[np.float16, -1, (2, 3, 4, 5)], [np.float16, -1, (2, 3, 4, 5)]],
            [[np.float16, -1, (4, 5, 6, 7)], [np.float16, -1, (4, 5, 6, 7)]],
            [[np.float16, -1, (4, 5, 6, 7)], [np.float16, -1, (2, 3, 4, 5)]],
            [[np.float16, -1, (2, 3, 4, 5, 6)], [np.float16, -1, (2, 3, 5, 6, 7)]]
        ]
        return shape_format

    def cpu_op_exec(self, input_data):
        output = torch.erfinv(input_data)
        output = output.numpy()
        return output

    def npu_op_exec(self, input_data):
        output = torch.erfinv(input_data)
        output = output.to("cpu").numpy()
        return output

    def npu_op_out_exec(self, input1, npu_out):
        torch.erfinv(input1, out=npu_out)
        output = npu_out.to("cpu").numpy()
        return output

    def cpu_op_exec_(self, input1):
        input1.erfinv_()
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        input1 = input1.to("npu")
        input1.erfinv_()
        output = input1.to("cpu").numpy()
        return output

    def test_erfinv_shape_format(self):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[1], -0.5, 0.5)
            if item[1][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            if item[1][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)

    def test_erfinv_out_shape_format(self):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -0.5, 0.5)
            _, npu_out = create_common_tensor(item[1], -0.5, 0.5)
            if item[0][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input, npu_out)
            if item[0][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output.shape, npu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)

    def test_erfinv__shape_format(self):
        shape_format = self.get_shapeFormat()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[1], -0.5, 0.5)
            if item[1][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec_(cpu_input)
            npu_output = self.npu_op_exec_(npu_input)
            if item[1][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)


if __name__ == "__main__":
    run_tests()
