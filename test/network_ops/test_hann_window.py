import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestHannWindow(TestCase):

    def cpu_op_exec(self, window_length, dtype):
        output = torch.hann_window(window_length, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec(self, window_length, dtype):
        output = torch.hann_window(window_length, dtype=dtype, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic(self, window_length, periodic, dtype):
        output = torch.hann_window(window_length, periodic, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_periodic(self, window_length, periodic, dtype):
        output = torch.hann_window(window_length, periodic, dtype=dtype, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def test_hann_window(self):
        shape_format = [
            [0, torch.float32],
            [1, torch.float32],
            [7, torch.float32],
            [12, torch.float32],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(*item)
            npu_output = self.npu_op_exec(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hann_window_periodic(self):
        shape_format = [
            [0, False, torch.float32],
            [1, False, torch.float32],
            [7, False, torch.float32],
            [12, False, torch.float32],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(*item)
            npu_output = self.npu_op_exec_periodic(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hann_window_float16(self):
        shape_format = [
            [0, torch.float16],
            [1, torch.float16],
            [7, torch.float16],
            [12, torch.float16],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(*item)
            npu_output = self.npu_op_exec(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hann_window_periodic_float16(self):
        shape_format = [
            [0, False, torch.float16],
            [1, False, torch.float16],
            [7, False, torch.float16],
            [12, False, torch.float16],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(*item)
            npu_output = self.npu_op_exec_periodic(*item)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
