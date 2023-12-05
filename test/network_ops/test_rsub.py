import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRsub(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.rsub(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.rsub(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        output = torch.rsub(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def rsub_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def rsub_scalar_result(self, shape_format):
        for item in shape_format:
            scalar = np.random.uniform(0, 100)
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, scalar)
            npu_output_scalar = self.npu_op_exec_scalar(npu_input1, scalar)

            self.assertRtolEqual(cpu_output, npu_output_scalar)

    def test_sub_shape_format_fp16_1d(self):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [32]], [np.float16, i, [32]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_1d(self):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [32]], [np.float16, i, [32]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp16_2d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [5, 3]], [np.float16, i, [5, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_2d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [5, 3]], [np.float16, i, [5, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp16_3d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [256, 480, 14]], [np.float16, i, [256, 480, 14]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_3d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [256, 480, 14]], [np.float16, i, [256, 480, 14]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp16_4d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]], [np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_fp32_4d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]], [np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    # int-------------------------------------------------------------------------------
    def test_sub_shape_format_int32_1d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [32]], [np.int32, i, [32]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_int32_2d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [5, 3]], [np.int32, i, [5, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_int32_3d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [256, 480, 14]], [np.int32, i, [256, 480, 14]]] for i in format_list]
        self.rsub_result(shape_format)

    def test_sub_shape_format_int32_4d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [32, 3, 3, 3]], [np.int32, i, [32, 3, 3, 3]]] for i in format_list]
        self.rsub_result(shape_format)

    # scalar----------------------------------------------------------------------------
    def test_sub_scalar_shape_format_fp16_1d(self):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [32]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_1d(self):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [32]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp16_2d(self):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_2d(self):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp16_3d(self):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_3d(self):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp16_4d(self):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128, 28]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    def test_sub_scalar_shape_format_fp32_4d(self):
        format_list = []
        shape_format = [[[np.float16, i, [32, 64, 128, 28]]] for i in format_list]
        self.rsub_scalar_result(shape_format)

    # byte scalar - tensor ----------------------------------------------------------------------------
    def test_scalar_sub_byte(self):
        s_cpu = torch.tensor([0, 1, 2, 3, 4]).byte()
        s_npu = s_cpu.npu()
        c_out = 1 - s_cpu
        n_out = 1 - s_npu
        self.assertRtolEqual(c_out.numpy(), n_out.cpu().numpy())


if __name__ == "__main__":
    run_tests()
