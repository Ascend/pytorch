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


class TestQuantizePerTensor(TestCase):

    def generate_data_per_tensor(self, min_d, max_d, shape_x, dtype_x):
        input_x = np.random.uniform(min_d, max_d, shape_x).astype(dtype_x)
        npu_input_x = torch.from_numpy(input_x)
        return npu_input_x

    def cpu_op_exec_per_tensor(self, input_x, input_scale, input_zero_point, dtype):
        output = torch.quantize_per_tensor(input_x, input_scale, input_zero_point, dtype).int_repr()
        output = output.numpy()
        return output

    def npu_op_exec_per_tensor(self, input_x, input_scale, input_zero_point, dtype):
        input_x = input_x.to("npu")
        output = torch.quantize_per_tensor(input_x, input_scale, input_zero_point, dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_per_tensor_3_3_0p1_10_int32(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint32)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_3_3_0p1_10_int8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (3, 3), np.float16)
        input_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec_per_tensor(input_cpu, 0.1, 10, torch.qint8)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_3_3_3_3_3_3_0p1_10_uint8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.1, 10, torch.quint8)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.1, 10, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_30_30_30_30_30_30_0p01_5_uint8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (30, 30, 30, 30, 30, 30), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.01, 5, torch.quint8)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.01, 5, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
