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
from torch_npu.testing.common_utils import create_common_tensor


class TestTanh(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.tanh(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.tanh(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_tanh_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, -1, (4, 3, 3)], 1, 100],
            [[np.float32, -1, (7, 5, 5)], 21474836, 21474837],
            [[np.float32, -1, (4, 44, 44)], 3450, 34020],
            [[np.float32, -1, (65500, 3, 3)], -214748, -214746],
            [[np.float32, -1, (1024, 448, 448)], 200, 300],
            [[np.float32, -1, (128, 3, 5)], 0.3219780311757745, 92],
            [[np.float32, -1, (8, 7, 7)], 0.4820305734500543, 28],
            [[np.float32, -1, (15, 8, 8)], 0.8563874665918477, 98],
            [[np.float32, -1, (11, 6, 6)], 0.0694198357720135, 50],
            [[np.float32, -1, (24, 24, 3)], -2, -2],
            [[np.float32, -1, (6, 10, 10)], 0.6447298684351989, 95],
            [[np.float32, -1, (3, 9, 9)], 0.8723538084975545, 85],
            [[np.float32, -1, (5, 5, 5)], 0.8283759153463854, 71],
            [[np.float32, -1, (5, 1, 1)], 0.24718684227306953, 25],
            [[np.float32, -1, (14, 7, 7)], 0.3989186243492233, 7],
            [[np.float32, -1, (4, 10, 10)], 0.7866457165672994, 5],
            [[np.float32, -1, (3, 7, 7)], 0.3793216987112159, 39],
            [[np.float32, -1, (2, 8, 8)], 0.9662927186969077, 5],
            [[np.float32, -1, (3, 7, 7)], 0.9956475043306917, 28],
            [[np.float32, -1, (7, 10, 10)], 0.769565434387681, 9],
            [[np.float32, -1, (54, 93, 3)], 0.6447298684351989, 95],
            [[np.float32, -1, (6, 3, 3)], 0.03133650248813469, 37],
            [[np.float32, -1, (65500, 1, 1)], 95, 100],
            [[np.float32, -1, (6, 3, 10)], 0.03133650248813469, 37],

        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tanh_float16_shape_format(self, device="npu"):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch.tanh(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (65500, 1)], 212, 225],
            [[np.float16, -1, (1024, 448, 448)], 200, 300],
            [[np.float16, -1, (16, 16)], -1000, -100],
            [[np.float16, -1, (4, 1)], -1.1754943508e-38, -1.1754943508e-38],
            [[np.float16, -1, (7, 5, 5)], 21474836, 21474837],
            [[np.float16, -1, (4, 44, 44)], 3450, 34020],
            [[np.float16, -1, (65500, 3, 3)], -214748, -214746],
            [[np.float16, -1, (64, 4, 4)], -9.313225746154785e-10, 9.313225746154785e-10],
            [[np.float16, -1, (128, 3, 5)],
             -0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508],
            [[np.float16, -1, (1, 1, 1)], 0.9283381566708346, 16],
            [[np.float16, -1, (6, 3, 10)], 0.03133650248813469, 37],
            [[np.float16, -1, (65500, 1, 1)], 95, 100],
            [[np.float16, -1, (13, 5, 5)], 0.9790231845699171, 41],
            [[np.float16, -1, (5, 7, 7)], 0.7852605507867441, 87],
            [[np.float16, -1, (13, 2, 2)], 0.8758750778305631, 82],
            [[np.float16, -1, (14, 6, 6)], 0.6963691068720794, 92],
            [[np.float16, -1, (5, 6, 6)], 0.7570129172808612, 21],
            [[np.float16, -1, (1, 10, 10)], 0.990800730328874, 86],
            [[np.float16, -1, (4, 5, 5)], 0.7349293532899402, 35],
            [[np.float16, -1, (6, 4, 4)], 0.7349293532899402, 35],
            [[np.float16, -1, (5, 8, 8)], 0.9583309378850908, 60],

        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = cpu_op_exec_fp16(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tanh_inplace_common_shape_format(self, device="npu"):
        def cpu_op_inplace_exec(input1):
            output = torch.tanh_(input1)
            output = output.numpy()
            return output

        def npu_op_inplace_exec(input1):
            input1 = input1.to("npu")
            output = torch.tanh_(input1)
            output = output.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
            [[np.float32, -1, (4, 3, 3)], 1, 100],
            [[np.float32, -1, (7, 5, 5)], 21474836, 21474837],
            [[np.float32, -1, (4, 44, 44)], 3450, 34020],
            [[np.float32, -1, (65500, 3, 3)], -214748, -214746]

        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output = cpu_op_inplace_exec(cpu_input1)
            npu_output = npu_op_inplace_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
