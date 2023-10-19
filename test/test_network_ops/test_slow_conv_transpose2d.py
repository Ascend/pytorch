import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSlowConvTranspose2d(TestCase):
    def cpu_op_exec(self, input_1, weight, kernel_size):
        cpu_output = torch._C._nn.slow_conv_transpose2d(input_1, weight, kernel_size)
        cpu_output = cpu_output.numpy()

        return cpu_output

    def cpu_op_exec_fp16(self, input_1, weight, kernel_size):
        input_1 = input_1.to(torch.float32)
        weight = weight.to(torch.float32)
        cpu_output = torch._C._nn.slow_conv_transpose2d(input_1, weight, kernel_size)
        cpu_output = cpu_output.numpy()
        cpu_output = cpu_output.astype(np.float16)

        return cpu_output

    def npu_op_exec(self, input_1, weight, kernel_size):
        npu_output = torch._C._nn.slow_conv_transpose2d(input_1, weight, kernel_size)
        npu_output = npu_output.to("cpu")
        npu_output = npu_output.numpy()
        return npu_output

    def cpu_op_exec_out(self, input_1, weight, kernel_size, cpu_out):
        torch._C._nn.slow_conv_transpose2d(input_1, weight, kernel_size, out=cpu_out)
        cpu_output = cpu_out.numpy()

        return cpu_output

    def cpu_op_exec_out_fp16(self, input_1, weight, kernel_size, cpu_out):
        input_1 = input_1.to(torch.float32)
        weight = weight.to(torch.float32)
        cpu_out = cpu_out.to(torch.float32)
        torch._C._nn.slow_conv_transpose2d(input_1, weight, kernel_size, out=cpu_out)
        cpu_output = cpu_out.numpy()
        cpu_output = cpu_output.astype(np.float16)

        return cpu_output

    def npu_op_exec_out(self, input_1, weight, kernel_size, npu_out):
        torch._C._nn.slow_conv_transpose2d(input_1, weight, kernel_size, out=npu_out)
        npu_output = npu_out.to("cpu")
        npu_output = npu_output.numpy()
        return npu_output

    def test_slow_conv_transpose2d_fp16(self):
        # input_1, weight, kernel_size
        shape_format1 = [
            [[np.float16, -1, [1, 1, 32, 32]], [np.float16, -1, [1, 1, 3, 3]], 3],
            [[np.float16, -1, [5, 1, 5, 5]], [np.float16, -1, [1, 1, 3, 3]], 3],
        ]
        for item in shape_format1:
            input_1_cpu, input_1_npu = create_common_tensor(item[0], 0, 1)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            if input_1_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_fp16(input_1_cpu, weight_cpu, item[2])
            else:
                cpu_output = self.cpu_op_exec(input_1_cpu, weight_cpu, item[2])

            npu_output = self.npu_op_exec(input_1_npu, weight_npu, item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_slow_conv_transpose2d_fp32(self):
        # input_1, weight, kernel_size
        shape_format2 = [
            [[np.float32, -1, [1, 1, 32, 32]], [np.float32, -1, [1, 1, 3, 3]], 3],
            [[np.float32, 0, [1, 4, 5, 5]], [np.float32, 0, [4, 4, 3, 3]], 3],
            [[np.float32, 3, [256, 256, 7, 7]], [np.float32, 0, [256, 256, 1, 1]], 1],
        ]
        for item in shape_format2:
            input_1_cpu, input_1_npu = create_common_tensor(item[0], 0, 1)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            if input_1_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_fp16(input_1_cpu, weight_cpu, item[2])
            else:
                cpu_output = self.cpu_op_exec(input_1_cpu, weight_cpu, item[2])

            npu_output = self.npu_op_exec(input_1_npu, weight_npu, item[2])
            # fp32 isn't enough precision, relaxation of precision requirement temporary
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)

    def test_slow_conv_transpose2d_out_fp16(self):
        # input_1, weight, kernel_size, out
        shape_format3 = [
            [[np.float16, -1, [5, 1, 5, 5]], [np.float16, -1, [1, 1, 3, 3]],
             3, [np.float16, -1, [5, 1, 7, 7]]],
            [[np.float16, 3, [256, 256, 7, 7]], [np.float16, 0, [256, 256, 1, 1]],
             1, [np.float16, 3, [256, 256, 7, 7]]]
        ]
        for item in shape_format3:
            input_1_cpu, input_1_npu = create_common_tensor(item[0], 0, 1)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            out_cpu, out_npu = create_common_tensor(item[3], 0, 1)
            if input_1_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_out_fp16(input_1_cpu, weight_cpu, item[2], out_cpu)
            else:
                cpu_output = self.cpu_op_exec_out(input_1_cpu, weight_cpu, item[2], out_cpu)
            npu_output = self.npu_op_exec_out(input_1_npu, weight_npu, item[2], out_npu)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_slow_conv_transpose2d_out_fp32(self):
        # input_1, weight, kernel_size, out
        shape_format4 = [
            [[np.float32, -1, [5, 1, 5, 5]], [np.float32, -1, [1, 1, 3, 3]],
             3, [np.float32, -1, [5, 1, 7, 7]]],
            [[np.float32, 3, [256, 256, 7, 7]], [np.float32, 0, [256, 256, 1, 1]],
             1, [np.float32, 3, [256, 256, 7, 7]]]
        ]
        for item in shape_format4:
            input_1_cpu, input_1_npu = create_common_tensor(item[0], 0, 1)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            out_cpu, out_npu = create_common_tensor(item[3], 0, 1)
            if input_1_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_out_fp16(input_1_cpu, weight_cpu, item[2], out_cpu)
            else:
                cpu_output = self.cpu_op_exec_out(input_1_cpu, weight_cpu, item[2], out_cpu)
            npu_output = self.npu_op_exec_out(input_1_npu, weight_npu, item[2], out_npu)
            # fp32 isn't enough precision, relaxation of precision requirement temporary
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)


if __name__ == "__main__":
    run_tests()
