import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestClip(TestCase):
    def generate_data(self, data):
        input1 = np.random.uniform(data[0], data[1], data[2]).astype(data[3])
        input1 = torch.from_numpy(input1)

        return input1

    def npu_op_exec(self, input1, min_val, max_val):
        input1 = input1.to("npu")
        output = torch.clip(input1, min_val, max_val)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def cpu_op_exec(self, input1, min_val, max_val):
        output = torch.clip(input1, min_val, max_val)
        output = output.numpy()

        return output

    def cpu_op_exec_float16(self, input1, min_val, max_val):
        input1 = input1.to(torch.float32)
        output = torch.clip(input1, min_val, max_val).to(torch.float16)
        output = output.numpy()

        return output

    def npu_inp_op_exec(self, input1, min_val, max_val):
        input1 = input1.to("npu")
        output = torch.clip_(input1, min_val, max_val)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def cpu_inp_op_exec(self, input1, min_val, max_val):
        output = torch.clip_(input1, min_val, max_val)
        output = output.numpy()

        return output

    def cpu_inp_op_exec_float16(self, input1, min_val, max_val):
        input1 = input1.to(torch.float32)
        output = torch.clip_(input1, min_val, max_val).to(torch.float16)
        output = output.numpy()

        return output

    def npu_op_exec_out(self, input1, min_val, max_val, input2):
        input1 = input1.to("npu")
        output = input2.to("npu")
        torch.clip(input1, min_val, max_val, out=output)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def npu_inp_uncon_op_exec(self, input1, min_val, max_val):
        input1 = input1.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clip_(input1, min_val, max_val)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def cpu_inp_uncon_op_exec(self, input1, min_val, max_val):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.clip(input1, min_val, max_val)
        output = output.numpy()

        return output

    def cpu_inp_uncon_op_exec_float16(self, input1, min_val, max_val):
        input1 = input1.to(torch.float32).as_strided([2, 2], [1, 2], 2)
        output = torch.clip(input1, min_val, max_val).to(torch.float16)
        output = output.numpy()

        return output

    def test_clip_common(self):
        shape_format = [
            [1, 100, (4, 3), np.float32],
            [1, 100, (4, 3), np.int32],
        ]
        for item in shape_format:
            input1 = self.generate_data(item)

            cpu_output = self.cpu_op_exec(input1, 40, 60)
            npu_output = self.npu_op_exec(input1, 40, 60)

            cpu_inp_output = self.cpu_inp_op_exec(input1, 40, 60)
            npu_inp_output = self.npu_inp_op_exec(input1, 40, 60)

            input2 = self.generate_data(item)
            npu_out_output = self.npu_op_exec_out(input1, 40, 60, input2)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec(input1, 40, 60)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input1, 40, 60)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)

    def test_clip_float16(self):
        shape_format = [
            [1, 100, (4, 3), np.float16],
        ]
        for item in shape_format:
            input1 = self.generate_data(item)

            cpu_output = self.cpu_op_exec_float16(input1, 40, 60)
            npu_output = self.npu_op_exec(input1, 40, 60)

            cpu_inp_output = self.cpu_inp_op_exec_float16(input1, 40, 60)
            npu_inp_output = self.npu_inp_op_exec(input1, 40, 60)

            input2 = self.generate_data(item)
            npu_out_output = self.npu_op_exec_out(input1, 40, 60, input2)

            cpu_inp_uncon_output = self.cpu_inp_uncon_op_exec_float16(input1, 40, 60)
            npu_inp_uncon_output = self.npu_inp_uncon_op_exec(input1, 40, 60)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_inp_output, npu_inp_output)
            self.assertRtolEqual(cpu_output, npu_out_output)
            self.assertRtolEqual(cpu_inp_uncon_output, npu_inp_uncon_output)


if __name__ == "__main__":
    run_tests()
