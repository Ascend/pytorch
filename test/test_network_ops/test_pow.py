import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestPow(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch.pow(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.pow(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.pow(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    '''
    currently do not support inplace op
    def cpu_op_inplace_exec(self, input1, input2):
        input1.pow_(input2)
        output = input1.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        input1.pow_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_tensor_scalar(self, input1, n):
        output = torch.pow(input1, n)
        output = output.numpy()
        return output

    def npu_op_exec_tensor_scalar(self, input1, n):
        output = torch.pow(input1, n)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor_scalar_out(self, input1, n, out):
        output = torch.pow(input1, n, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_tensor(self, n, input1):
        output = torch.pow(n, input1)
        output = output.numpy()
        return output

    def npu_op_exec_scalar_tensor(self, n, input1):
        output = torch.pow(n, input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_tensor_out(self, n, input1, out):
        torch.pow(n, input1, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output


    currently do not support scalar dtype
    def pow_result_scalar_tensor(self, shape_format):
        for item in shape_format:
            scalar = np.random.randint(0, 1)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 1)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_scalar = self.cpu_op_exec_scalar_tensor(scalar, cpu_input1)
            npu_output_scalar = self.npu_op_exec_scalar_tensor(scalar, npu_input1)
            npu_output_scalar_out = self.npu_op_exec_scalar_tensor_out(scalar, npu_input1, npu_input3)

            cpu_output_scalar = cpu_output_scalar.astype(npu_output_scalar.dtype)
            self.assertRtolEqual(cpu_output_scalar, npu_output_scalar)
            self.assertRtolEqual(cpu_output_scalar, npu_output_scalar_out)

    def pow_result_tensor_scalar_(self, shape_format):
        for item in shape_format:
            scalar = np.random.randint(0, 1)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 1)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_tensor_scalar = self.cpu_op_exec_tensor_scalar(cpu_input1, scalar)
            npu_output_tensor_scalar = self.npu_op_exec_tensor_scalar(npu_input1, scalar)
            npu_output_tensor_scalar_out = self.npu_op_exec_tensor_scalar_out(npu_input1, scalar, npu_input3)

            cpu_output_tensor_scalar = cpu_output_tensor_scalar.astype(npu_output_tensor_scalar.dtype)
            self.assertRtolEqual(cpu_output_tensor_scalar, npu_output_tensor_scalar)
            self.assertRtolEqual(cpu_output_tensor_scalar, npu_output_tensor_scalar_out)

    # scalar_tensor-------------------------------------------------------
    def test_pow_shape_format_scalar_tensor_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    def test_pow_shape_format_scalar_tensor_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    def test_pow_shape_format_scalar_tensor_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    def test_pow_shape_format_scalar_tensor_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    def test_pow_shape_format_scalar_tensor_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64, 128]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    def test_pow_shape_format_scalar_tensor_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64, 128]] for i in format_list]
        self.pow_result_scalar_tensor(shape_format)

    # tensor_scalar-----------------------------------------------------------
    def test_pow_shape_format_tensor_scala_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    def test_pow_shape_format_tensor_scalar_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    def test_pow_shape_format_tensor_scala_fp16_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    def test_pow_shape_format_tensor_scalar_fp32_2d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    def test_pow_shape_format_tensor_scala_fp16_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float16, i, [18, 64, 128]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)

    def test_pow_shape_format_tensor_scalar_fp32_3d(self, device):
        format_list = [-1, 0, 3, 29]
        shape_format = [[np.float32, i, [18, 64, 128]] for i in format_list]
        self.pow_result_tensor_scalar_(shape_format)
    '''

    # tensor_tensor-----------------------------------------------------------
    def test_pow_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (5, )], [np.float32, -1, (1, )]],
            [[np.float32, -1, (4, 3)], [np.float32, -1, (4, 1)]],
            [[np.float32, -1, (4, 3, 1)], [np.float32, -1, (4, 1, 5)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 0, 1)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_pow_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.pow(input1, input2)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (5, )], [np.float16, -1, (1, )]],
            [[np.float16, -1, (4, 3)], [np.float16, -1, (4, 1)]],
            [[np.float16, -1, (4, 3, 1)], [np.float16, -1, (4, 1, 5)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 2)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 0, 2)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_pow_int32_float_format(self):
        a = torch.randn(64).to(torch.int32)
        na = a.npu()
        cpu_out = self.cpu_op_exec(a, 1.0)
        npu_out = self.npu_op_exec(na, 1.0)
        self.assertRtolEqual(cpu_out, npu_out)

    def test_pow_float_int32_format(self):
        a = torch.randn(64).to(torch.int32)
        na = a.npu()
        cpu_out = self.cpu_op_exec(1.0, a)
        npu_out = self.npu_op_exec(1.0, na)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
