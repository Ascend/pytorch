# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
from torch_npu.testing.common_utils import create_common_tensor


class TestViewAs(TestCase):

    def cpu_op_real_exec(self, input1):
        real = input1.numpy().real
        imag = input1.numpy().imag
        return np.stack(arrays=(real, imag), axis=-1)

    def npu_op_real_exec(self, input1):
        output = torch.view_as_real(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_complex_exec(self, input1):
        return input1.real.numpy(), input1.imag.numpy()

    def npu_op_complex_exec(self, input1):
        output1 = torch.view_as_real(input1)
        output2 = torch.view_as_complex(output1)
        output = output2.to("cpu")
        return output.real.numpy(), output.imag.numpy()

    def test_view_as_real(self):
        shape_format = [
            [[np.complex64, 0, (5, 3, 6, 4)]],
            [[np.complex128, 0, (5, 3, 6, 4)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_real_exec(cpu_input1)
            npu_output = self.npu_op_real_exec(npu_input1)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_view_as_complex(self):
        shape_format = [
            [[np.complex64, 0, (5, 3, 6, 4)]],
            [[np.complex128, 0, (5, 3, 6, 4)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output_real, cpu_output_complex = self.cpu_op_complex_exec(cpu_input1)
            npu_output_real, npu_output_complex = self.npu_op_complex_exec(npu_input1)

            self.assertRtolEqual(cpu_output_real, npu_output_real)
            self.assertRtolEqual(cpu_output_complex, npu_output_complex)


if __name__ == "__main__":
    run_tests()
