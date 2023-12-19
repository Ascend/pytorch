import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestHammingWindow(TestCase):

    def cpu_op_exec(self, window_length, dtype):
        output = torch.hamming_window(window_length, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec(self, window_length, dtype):
        output = torch.hamming_window(window_length, dtype=dtype, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic(self, window_length, periodic, dtype):
        output = torch.hamming_window(window_length, periodic, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_periodic(self, window_length, periodic, dtype):
        output = torch.hamming_window(window_length, periodic, dtype=dtype, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic_alpha(self, window_length, periodic, alpha, dtype):
        output = torch.hamming_window(window_length, periodic, alpha, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_periodic_alpha(self, window_length, periodic, alpha, dtype):
        output = torch.hamming_window(window_length, periodic, alpha, dtype=dtype, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic_alpha_beta(self, window_length, periodic, alpha, beta, dtype):
        output = torch.hamming_window(window_length, periodic, alpha, beta, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_periodic_alpha_beta(self, window_length, periodic, alpha, beta, dtype):
        output = torch.hamming_window(window_length, periodic, alpha, beta, dtype=dtype, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def test_hamming_window(self):
        shape_format = [
            [0, torch.float32],
            [1, torch.float32],
            [7, torch.float32],
            [12, torch.float32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(*item)
            npu_output = self.npu_op_exec(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_periodic(self):
        shape_format = [
            [0, False, torch.float32],
            [1, False, torch.float32],
            [7, False, torch.float32],
            [12, False, torch.float32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(*item)
            npu_output = self.npu_op_exec_periodic(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_periodic_alpha(self):
        shape_format = [
            [0, True, 0.22, torch.float32],
            [0, True, 2.2, torch.float32],
            [1, True, 0.22, torch.float32],
            [1, True, 2.0, torch.float32],
            [7, True, 0.22, torch.float32],
            [7, True, 2.0, torch.float32],
            [12, True, 0.22, torch.float32],
            [12, True, 2.0, torch.float32],
            [0, False, 0.22, torch.float32],
            [0, False, 2.2, torch.float32],
            [1, False, 2.0, torch.float32],
            [7, False, 2.0, torch.float32],
            [12, False, 1.1, torch.float32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic_alpha(*item)
            npu_output = self.npu_op_exec_periodic_alpha(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hammingwindow_periodic_alpha_beta(self):
        shape_format = [
            [0, True, 0.44, 0.22, torch.float32],
            [1, True, 0.44, 0.22, torch.float32],
            [7, True, 0.44, 0.22, torch.float32],
            [12, True, 0.44, 0.22, torch.float32],
            [7, True, 4.4, 2.2, torch.float32],
            [1, True, 4.4, 2.2, torch.float32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic_alpha_beta(*item)
            npu_output = self.npu_op_exec_periodic_alpha_beta(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_float16(self):
        shape_format = [
            [0, torch.float16],
            [1, torch.float16],
            [7, torch.float16],
            [12, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(*item)
            npu_output = self.npu_op_exec(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_periodic_float16(self):
        shape_format = [
            [0, False, torch.float16],
            [1, False, torch.float16],
            [7, False, torch.float16],
            [12, False, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(*item)
            npu_output = self.npu_op_exec_periodic(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_periodic_alpha_float16(self):
        shape_format = [
            [0, True, 0.22, torch.float16],
            [0, True, 2.2, torch.float16],
            [1, True, 0.22, torch.float16],
            [1, True, 2.0, torch.float16],
            [7, True, 0.22, torch.float16],
            [7, True, 2.0, torch.float16],
            [12, True, 0.22, torch.float16],
            [12, True, 2.0, torch.float16],
            [0, False, 0.22, torch.float16],
            [0, False, 2.2, torch.float16],
            [1, False, 2.0, torch.float16],
            [7, False, 2.0, torch.float16],
            [12, False, 1.1, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic_alpha(*item)
            npu_output = self.npu_op_exec_periodic_alpha(*item)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hammingwindow_periodic_alpha_beta_float16(self):
        shape_format = [
            [0, True, 0.44, 0.22, torch.float16],
            [1, True, 0.44, 0.22, torch.float16],
            [7, True, 0.44, 0.22, torch.float16],
            [12, True, 0.44, 0.22, torch.float16],
            [7, True, 4.4, 2.2, torch.float16],
            [1, True, 4.4, 2.2, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic_alpha_beta(*item)
            npu_output = self.npu_op_exec_periodic_alpha_beta(*item)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
