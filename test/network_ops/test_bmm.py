import itertools
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchMatMul(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch.bmm(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.bmm(input1, input2)
        output = output.to("cpu").numpy()
        return output

    def npu_op_out_exec(self, input1, input2, output):
        torch.bmm(input1, input2, out=output)
        output = output.to("cpu").numpy()
        return output

    def bmm_auto_list_exec(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

    def bmm_out_op_exec(self, shape):
        for item in shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            _, npu_output = create_common_tensor(item[1], 0, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, npu_output)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertEqual(cpu_output.shape, npu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)

    def test_bmm_out_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(1, 3, 2)]
        shape_format1 = [[np.float16, i, j] for i in format_list for j in shape_list]
        shape_list = [(1, 2, 3)]
        shape_format2 = [[np.float16, i, j] for i in format_list for j in shape_list]
        shape_format = [[i, j] for i in shape_format1 for j in shape_format2]
        self.bmm_out_op_exec(shape_format)

    def test_bmm_out_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(1, 3, 2)]
        shape_format1 = [[np.float32, i, j] for i in format_list for j in shape_list]
        shape_list = [(1, 2, 3)]
        shape_format2 = [[np.float32, i, j] for i in format_list for j in shape_list]
        shape_format = [[i, j] for i in shape_format1 for j in shape_format2]
        self.bmm_out_op_exec(shape_format)

    def test_bmm_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(1, 3, 2)]
        shape_format1 = [[np.float16, i, j] for i in format_list for j in shape_list]
        shape_list = [(1, 2, 3)]
        shape_format2 = [[np.float16, i, j] for i in format_list for j in shape_list]
        shape_format = [[i, j] for i in shape_format1 for j in shape_format2]
        self.bmm_auto_list_exec(shape_format)

    def test_bmm_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_list = [(1, 3, 2)]
        shape_format1 = [[np.float32, i, j] for i in format_list for j in shape_list]
        shape_list = [(1, 2, 3)]
        shape_format2 = [[np.float32, i, j] for i in format_list for j in shape_list]
        shape_format = [[i, j] for i in shape_format1 for j in shape_format2]
        self.bmm_auto_list_exec(shape_format)


if __name__ == "__main__":
    run_tests()
