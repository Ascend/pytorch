import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBitwiseOr(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_bool_data(self, min_d, max_d, shape):
        input1 = np.random.uniform(min_d, max_d, shape)
        input2 = np.random.uniform(min_d, max_d, shape)
        input1 = input1.reshape(-1)
        input2 = input2.reshape(-1)
        seq = range(len(input1))
        for i in enumerate(seq):
            if input1[i] < 0.5:
                input1[i] = 0
        seq = range(len(input1))
        for i in enumerate(seq):
            if input2[i] < 0.5:
                input2[i] = 0
        input1 = input1.astype(np.bool_)
        input2 = input2.astype(np.bool_)
        input1 = input1.reshape(shape)
        input2 = input2.reshape(shape)
        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_single_bool_data(self, min_d, max_d, shape):
        input1 = np.random.uniform(min_d, max_d, shape)
        input1 = input1.reshape(-1)
        seq = range(len(input1))
        for i in enumerate(seq):
            if input1[i] < 0.5:
                input1[i] = 0
        input1 = input1.astype(np.bool_)
        input1 = input1.reshape(shape)
        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def npu_op_exec_out(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input3.to("npu")
        torch.bitwise_or(input1, input2, out=output)
        output = output.to("cpu")
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def npu_mix_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.bitwise_or(input1, input2)
        output = output.to("cpu")
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1, input2):
        output = torch.bitwise_or(input1, input2)
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, output):
        torch.bitwise_or(input1, input2, out=output)
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def cpu_op_exec_scalar(self, input1, scalar):
        output = torch.bitwise_or(input1, scalar)
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_out(self, input1, scalar, output):
        torch.bitwise_or(input1, scalar, out=output)
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def npu_op_exec_scalar_out(self, input1, scalar, output):
        input1 = input1.to("npu")
        output = output.to("npu")
        output = torch.bitwise_or(input1, scalar, out=output)
        output = output.to("cpu")
        if output.dtype not in [torch.int32, torch.bool]:
            output = output.to(torch.int32)
        output = output.numpy()
        return output

    def bitwise_or_tensor_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[1], 0, 100)
            cpu_output_out = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)

            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_bitwise_or_tensor_out(self, device="npu"):
        shape_format = [
            [[np.int16, 0, [128, 3, 224, 224]], [np.int16, 0, [3, 3, 3]]],
            [[np.int16, 0, [128, 116, 14, 14]], [np.int16, 0, [128, 116, 14, 14]]],
            [[np.int32, 0, [256, 128, 7, 7]], [np.int32, 0, [128, 256, 3, 3]]],
            [[np.int32, 0, [2, 3, 3, 3]], [np.int32, 0, [3, 1, 3]]],
            [[np.int32, 0, [128, 232, 7, 7]], [np.int32, 0, [128, 232, 7, 7]]],
        ]
        self.bitwise_or_tensor_out_result(shape_format)

    def bitwise_or_scalar_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            scalar = np.random.randint(1, 5)
            cpu_output_out = self.cpu_op_exec_scalar(cpu_input1, scalar)
            npu_output_out = self.npu_op_exec_scalar_out(npu_input1, scalar, npu_input2)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_bitwise_or_scalar_out(self, device="npu"):
        shape_format = [
            [[np.int16, 0, [16, 3, 1111, 1212]], [np.int16, 0, [3, 3, 3]]],
            [[np.int16, 0, [128, 116, 14, 14]], [np.int16, 0, [128, 116, 14, 14]]],
            [[np.int32, 0, [1313, 3, 3, 3]], [np.int32, 0, [3, 1, 3]]],
            [[np.int32, 0, [128, 232, 7, 7]], [np.int32, 0, [128, 232, 7, 7]]],
        ]
        self.bitwise_or_scalar_out_result(shape_format)

    def test_bitwise_or_int32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input1)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_bool_scalar(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec_out(npu_input1, True, npu_input1)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, True, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_int32_scalar(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec_out(npu_input1, 1, npu_input1)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 1, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_int16(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (1, 6), np.int16)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input1)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_int16_scalar(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int16)
        cpu_output = self.cpu_op_exec_out(npu_input1, 1, npu_input1)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 1, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_int16_diff(self, device="npu"):
        npu_input1 = self.generate_single_data(0, 100, (1, 6), np.int16)
        npu_input2 = self.generate_single_data(0, 100, (1, 1), np.int16)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input1)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_int16_out(self, device="npu"):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 100, (4, 3), np.int16)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_bitwise_or_mix_dtype(self, device="npu"):
        npu_input1 = self.generate_single_data(0, 100, (1, 6), np.int32)
        npu_input2 = self.generate_single_data(0, 100, (1, 6), np.int16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_mix_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
