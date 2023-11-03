import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFix(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.fix(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.fix(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_exec(self, input1):
        input1.fix_()
        output = input1.numpy()
        return output

    def npu_op_inp_exec(self, input1):
        input1.fix_()
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2):
        torch.trunc(input1, out=input2)
        output = input2.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.trunc(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def test_fix_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (4, 3, 1)]],
            [[np.float32, -1, (2, 3)]],
            [[np.float32, -1, (2, 3, 4, 5)]],
            [[np.float32, -1, (10,)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fix_inp_float32_format(self):
        shape_format = [
            [[np.float32, -1, (4, 3, 1)]],
            [[np.float32, -1, (2, 3)]],
            [[np.float32, -1, (2, 3, 4, 5)]],
            [[np.float32, -1, (10,)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_output = self.cpu_op_inp_exec(cpu_input1)
            npu_output = self.npu_op_inp_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fix_inp_float16_format(self):
        shape_format = [
            [[np.float16, -1, (4, 3, 1)]],
            [[np.float16, -1, (2, 3)]],
            [[np.float16, -1, (2, 3, 4, 5)]],
            [[np.float16, -1, (10,)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_inp_exec(cpu_input1)
            npu_output = self.npu_op_inp_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_add_float16_shape_format(self):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch.fix(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (2, 3)]],
            [[np.float16, -1, (4, 3, 1)]],
            [[np.float16, -1, (2, 3, 4, 5)]],
            [[np.float16, -1, (10,)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_output = cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fix_out_format(self):
        shape_format = [
            [[np.float16, -1, (4, 3, 1)]],
            [[np.float16, -1, (2, 3)]],
            [[np.float16, -1, (2, 3, 4, 5)]],
            [[np.float16, -1, (10,)]],
            [[np.float32, -1, (4, 3, 1)]],
            [[np.float32, -1, (2, 3)]],
            [[np.float32, -1, (2, 3, 4, 5)]],
            [[np.float32, -1, (10,)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -5, 5)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -5, 5)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
