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


class TestQuantizePerChannel(TestCase):
    def generate_data_per_channel(self, min_d, max_d, shape_x, shape_scale, shape_zp, dtype_x, dtype_scale, dtype_zp):
        input_x = np.random.uniform(min_d, max_d, shape_x).astype(dtype_x)
        scales = np.random.uniform(min_d, max_d, shape_scale).astype(dtype_scale)
        zero_points = np.random.uniform(min_d, max_d, shape_zp).astype(dtype_zp)
        npu_input_x = torch.from_numpy(input_x)
        npu_input_scales = torch.from_numpy(scales)
        npu_input_zero_points = torch.from_numpy(zero_points)
        return npu_input_x, npu_input_scales, npu_input_zero_points

    def cpu_op_exec_per_channel(self, input_x, input_scales, input_zero_points, axis, dtype):
        output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype).int_repr()
        output = output.numpy()
        return output

    def npu_op_exec_per_channel(self, input_x, input_scales, input_zero_points, axis, dtype):
        input_x = input_x.to("npu")
        input_scales = input_scales.to("npu")
        input_zero_points = input_zero_points.to("npu")
        output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_per_channel_3_3_0_int32(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (3, 3), (3,), (3,), np.float32,
                                                                       np.float32, np.int32)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 0, torch.qint32)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 0, torch.qint32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_3_3_3_3_1_int8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (3, 3), (3,), (3,), np.float32,
                                                                       np.float32, np.int8)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 1, torch.qint8).astype(np.int32)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 1, torch.qint8).astype(np.int32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_3_3_3_3_3_3_3_3_4_uint8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (3, 3, 3, 3, 3, 3, 3, 3), (3,), (3,),
                                                                       np.float32, np.float32, np.int32)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 4, torch.quint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 4, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_30_30_30_30_30_2_uint8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (30, 30, 30, 30), (30,), (30,),
                                                                       np.float16, np.float32, np.uint8)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1_cpu, scales, zero_points, 2, torch.quint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 2, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
