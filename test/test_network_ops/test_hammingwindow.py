# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestHammingWindow(TestCase):
    def test_hammingwindow(self):
        shape_format = [
            [7, True, 0.44, 0.22, torch.float32],
            [10, False, 0.44, 0.22, torch.float32]]

        for item in shape_format:
            cpu_output = torch.hamming_window(item[0], item[1], item[2], item[3], dtype=item[4]).numpy()
            npu_output = torch.hamming_window(item[0], item[1], item[2], item[3], dtype=item[4]).cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def generate_output_data(self, min1, max1, shape, dtype):
        output_y = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_output_y = torch.from_numpy(output_y)
        return npu_output_y

    def cpu_op_exec_out(self, window_length, periodic, alpha, beta, dtype, output_y):
        output = output_y
        torch.hamming_window(window_length, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, out=output_y)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, window_length, periodic, alpha, beta, dtype, output_y):
        output = output_y.to("npu")
        torch.hamming_window(window_length, periodic=periodic, alpha=alpha, beta=beta,
                             dtype=dtype, out=output_y, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output


if __name__ == "__main__":
    run_tests()
