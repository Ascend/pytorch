import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNegative(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.negative(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.negative(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.negative(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        torch.negative_(input1)
        output = input1.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        torch.negative_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def negative_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_input_inp, npu_input_inp = create_common_tensor(item[0], -100, 100)
            if cpu_input_inp.dtype == torch.float16:
                cpu_input_inp = cpu_input_inp.to(torch.float32)
            cpu_output_inp = self.cpu_inp_op_exec(cpu_input_inp)
            npu_output_inp = self.npu_inp_op_exec(npu_input_inp)
            cpu_output_inp = cpu_output_inp.astype(npu_output_inp.dtype)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def negative_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[1], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output_out1 = self.npu_op_exec_out(npu_input1, npu_input2)
            npu_output_out2 = self.npu_op_exec_out(npu_input1, npu_input3)
            cpu_output = cpu_output.astype(npu_output_out1.dtype)
            self.assertRtolEqual(cpu_output, npu_output_out1)
            self.assertRtolEqual(cpu_output, npu_output_out2)

    def test_negative_out_result(self):
        shape_format = [
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [128, 58, 28, 28]], [np.float16, 0, [58, 58, 1, 1]]],
            [[np.float16, 0, [128, 3, 224, 224]], [np.float16, 0, [3, 3, 3, 3]]],
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [116, 116, 1, 1]]],
            [[np.float32, 0, [256, 128, 7, 7]], [np.float32, 0, [128, 128, 3, 3]]],
            [[np.float32, 0, [256, 3, 224, 224]], [np.float32, 0, [3, 3, 7, 7]]],
            [[np.float32, 0, [2, 3, 3, 3]], [np.float32, 0, [3, 1, 3, 3]]],
            [[np.float32, 0, [128, 232, 7, 7]], [np.float32, 0, [232, 232, 1, 1]]],
        ]
        self.negative_out_result(shape_format)

    def test_negative_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [[[np.float16, i, [96]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [[[np.float32, i, [96]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float16, i, [448, 1]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float32, i, [448, 1]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float16, i, [64, 24, 38]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float32, i, [64, 24, 38]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.negative_result(shape_format)

    def test_negative_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float32, i, [32, 3, 3, 3]]] for i in format_list]
        self.negative_result(shape_format)


if __name__ == "__main__":
    run_tests()
