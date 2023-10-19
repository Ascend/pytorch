# Copyright (c) 2020 Huawei Technologies Co., Ltd

# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestHammingWindow(TestCase):

    def cpu_op_exec(self, window_length):
        output = torch.hamming_window(window_length)
        output = output.numpy()
        return output

    def npu_op_exec(self, window_length):
        output = torch.hamming_window(window_length, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic(self, window_length, periodic):
        output = torch.hamming_window(window_length, periodic)
        output = output.numpy()
        return output

    def npu_op_exec_periodic(self, window_length, periodic):
        output = torch.hamming_window(window_length, periodic, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic_alpha(self, window_length, periodic, alpha):
        output = torch.hamming_window(window_length, periodic, alpha)
        output = output.numpy()
        return output

    def npu_op_exec_periodic_alpha(self, window_length, periodic, alpha):
        output = torch.hamming_window(window_length, periodic, alpha, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic_alpha_beta(self, window_length, periodic, alpha, beta):
        output = torch.hamming_window(window_length, periodic, alpha, beta)
        output = output.numpy()
        return output

    def npu_op_exec_periodic_alpha_beta(self, window_length, periodic, alpha, beta):
        output = torch.hamming_window(window_length, periodic, alpha, beta, device='npu')
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
            cpu_output = self.cpu_op_exec(item[0])
            npu_output = self.npu_op_exec(item[0])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_periodic(self):
        shape_format = [
            [0, False, torch.float32],
            [1, False, torch.float32],
            [7, False, torch.float32],
            [12, False, torch.float32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(item[0], item[1])
            npu_output = self.npu_op_exec_periodic(item[0], item[1])
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
            cpu_output = self.cpu_op_exec_periodic_alpha(item[0], item[1], item[2])
            npu_output = self.npu_op_exec_periodic_alpha(item[0], item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hammingwindow_periodic_alpha_beta(self):
        shape_format = [
            [0, True, 0.44, 0.22, torch.float32],
            [1, True, 0.44, 0.22, torch.float32],
            [7, True, 0.44, 0.22, torch.float32],
            [12, True, 0.44, 0.22, torch.float32],
            [0, False, 0.44, 0.22, torch.int32],
            [1, False, 0.44, 0.22, torch.int32],
            [7, False, 0.44, 0.22, torch.int32],
            [12, False, 0.44, 0.22, torch.int32],
            [7, True, 4.4, 2.2, torch.float32],
            [1, True, 4.4, 2.2, torch.float32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic_alpha_beta(item[0], item[1], item[2], item[3])
            npu_output = self.npu_op_exec_periodic_alpha_beta(item[0], item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_float16(self):
        shape_format = [
            [0, torch.float16],
            [1, torch.float16],
            [7, torch.float16],
            [12, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0])
            npu_output = self.npu_op_exec(item[0])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hamming_window_periodic_float16(self):
        shape_format = [
            [0, False, torch.float16],
            [1, False, torch.float16],
            [7, False, torch.float16],
            [12, False, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(item[0], item[1])
            npu_output = self.npu_op_exec_periodic(item[0], item[1])
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
            cpu_output = self.cpu_op_exec_periodic_alpha(item[0], item[1], item[2])
            npu_output = self.npu_op_exec_periodic_alpha(item[0], item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hammingwindow_periodic_alpha_beta_float16(self):
        shape_format = [
            [0, True, 0.44, 0.22, torch.float16],
            [1, True, 0.44, 0.22, torch.float16],
            [7, True, 0.44, 0.22, torch.float16],
            [12, True, 0.44, 0.22, torch.float16],
            [0, False, 0.44, 0.22, torch.int32],
            [1, False, 0.44, 0.22, torch.int32],
            [7, False, 0.44, 0.22, torch.int32],
            [12, False, 0.44, 0.22, torch.int32],
            [7, True, 4.4, 2.2, torch.float16],
            [1, True, 4.4, 2.2, torch.float16]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic_alpha_beta(item[0], item[1], item[2], item[3])
            npu_output = self.npu_op_exec_periodic_alpha_beta(item[0], item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
