import numpy as np

import torch
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSoftMax(TestCase):
    def cpu_op_exec(self, input1, dim):
        output = torch.nn.functional.softmax(input1, dim)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim):
        output = torch.nn.functional.softmax(input1, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_dtype(self, input1, dim, dtype):
        output = torch.nn.functional.softmax(input1, dim, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_dtype(self, input1, dim, dtype):
        output = torch.nn.functional.softmax(input1, dim, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_half_float(self, input1, dim):
        output = torch._softmax(input1, dim, True)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def softmax_result(self, shape_format):
        for item in shape_format:
            dim = np.random.randint(0, len(item[2]))
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_exec(npu_input1, dim)

            if npu_input1.dtype == torch.float16:
                npu_output_half = self.npu_op_exec_half_float(npu_input1, dim)
                npu_output_half = npu_output_half.astype(np.float16)

            cpu_output_inp = self.cpu_op_exec_dtype(cpu_input1, dim, torch.float32)
            npu_output_inp = self.npu_op_exec_dtype(npu_input1, dim, torch.float32)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_inp = cpu_output_inp.astype(npu_output_inp.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            if npu_input1.dtype == torch.float16:
                self.assertRtolEqual(cpu_output, npu_output_half)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def test_softmax_shape_format_fp16_1d(self, device='npu'):
        format_list = [0]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp16_2d(self, device='npu'):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [5, 256]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp16_3d(self, device='npu'):
        format_list = [0, 29]
        shape_format = [[np.float16, i, [32, 8, 8]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp16_4d(self, device='npu'):
        format_list = [0, 29]
        shape_format = [[np.float16, i, [64, 112, 7, 7]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp32_1d(self, device='npu'):
        format_list = [0]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp32_2d(self, device='npu'):
        format_list = [3, 29]
        shape_format = [[np.float32, i, [5, 256]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp32_3d(self, device='npu'):
        format_list = [0, 29]
        shape_format = [[np.float32, i, [32, 3, 3]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_shape_format_fp32_4d(self, device='npu'):
        format_list = [3, 29]
        shape_format = [[np.float32, i, [64, 112, 7, 7]] for i in format_list]
        self.softmax_result(shape_format)

    def test_softmax_dimname_shape_format(self, device='npu'):
        cpu_input1 = torch.randn(4, 3, names=('N', 'C'))
        npu_input1 = cpu_input1.npu()
        cpu_output = self.cpu_op_exec(cpu_input1, 'N')
        npu_output = self.npu_op_exec(npu_input1, 'N')
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
