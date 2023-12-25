import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestLog(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.log(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.log(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1):
        input1 = input1.to("npu")
        output = input1.to("npu")
        torch.log(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        output = torch.log_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.log_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log_(input1)
        output = output.numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1):
        input1 = input1.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_log_shape_format_fp32(self, device="npu"):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_shape_format_fp16(self, device="npu"):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_shape_format_fp32(self, device="npu"):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_shape_format_fp16(self, device="npu"):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_uncon_shape_format_fp32(self, device="npu"):
        format_list = [3]
        shape_list = [(8, 6)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_inp_uncon_op_exec(cpu_input1)
            npu_output = self.npu_inp_uncon_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_uncon_shape_format_fp16(self, device="npu"):
        format_list = [3]
        shape_list = [(8, 6)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_uncon_op_exec(cpu_input1)
            npu_output = self.npu_inp_uncon_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
