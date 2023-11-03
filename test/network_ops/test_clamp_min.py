import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestClampMin(TestCase):

    def npu_op_exec(self, input1, min_val):
        output = torch.clamp_min(input1, min_val)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec(self, input1, min_val):
        input_dtype = input1.dtype
        if input_dtype == torch.float16:
            input1 = input1.to(torch.float32)
        output = torch.clamp_min(input1, min_val)
        if input_dtype == torch.float16:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1, min_val):
        torch.clamp_min_(input1, min_val)
        output = input1.cpu().numpy()
        return output

    def cpu_inp_op_exec(self, input1, min_val):
        input_dtype = input1.dtype
        if input_dtype == torch.float16:
            input1 = input1.to(torch.float32)
        output = torch.clamp_min_(input1, min_val)
        if input_dtype == torch.float16:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, min_val, out_npu):
        torch.clamp_min(input1, min_val, out=out_npu)
        output = out_npu.cpu().numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1, min_val):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        torch.clamp_min_(input1, min_val)
        output = input1.cpu().numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1, min_val):
        input_dtype = input1.dtype
        if input_dtype == torch.float16:
            input1 = input1.to(torch.float32)
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clamp_min(input1, min_val)
        if input_dtype == torch.float16:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def test_clamp_min_common(self):
        shape_format = [
            [np.float32, 0, (4, 3)],
            [np.int32, 0, (4, 3)],
            [np.int64, 0, (4, 3)],
            [np.float16, 0, (4, 3)]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item, 1, 100)
            _, out_npu = create_common_tensor(item, 1, 100)

            cpu_output = self.cpu_op_exec(input_cpu, 50)
            npu_output = self.npu_op_exec(input_npu, 50)

            cpu_inp_output = self.cpu_inp_op_exec(input_cpu, 50)
            npu_inp_output = self.npu_inp_op_exec(input_npu, 50)

            npu_out_output = self.npu_op_exec_out(input_npu, 50, out_npu)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec(input_cpu, 50)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input_npu, 50)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clamp_min_tensor(self):
        shape_format = [
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
            [[np.int32, 0, (24, 13)], [np.int32, 0, (1, 13)]],
            [[np.int64, 0, (41, 32, 23)], [np.int32, 0, (41, 32, 23)]],
            [[np.float16, 0, (14, 3)], [np.float32, 0, (14, 3)]]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 1, 100)
            min_cpu, min_npu = create_common_tensor(item[1], 1, 50)
            _, out_npu = create_common_tensor(item[0], 1, 100)

            cpu_output = self.cpu_op_exec(input_cpu, min_cpu)
            npu_output = self.npu_op_exec(input_npu, min_npu)

            cpu_inp_output = self.cpu_inp_op_exec(input_cpu, min_cpu)
            npu_inp_output = self.npu_inp_op_exec(input_npu, min_npu)

            npu_out_output = self.npu_op_exec_out(input_npu, min_npu, out_npu)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)


if __name__ == "__main__":
    run_tests()
