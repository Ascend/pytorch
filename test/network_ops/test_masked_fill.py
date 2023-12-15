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


class TestMaskedFill(TestCase):
    def create_bool_tensor(self, shape, minValue, maxValue):
        input1 = np.random.uniform(minValue, maxValue, shape)
        input1 = input1 > 0.5
        cpu_input = torch.from_numpy(input1)
        npu_input = torch.from_numpy(input1).to("npu")
        return cpu_input, npu_input

    def cpu_op_exec(self, input1, mask, value):
        output = torch.masked_fill(input1, mask, value)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, mask, value):
        if torch.is_tensor(value) and value.device.type != 'npu':
            value = value.to("npu")
        output = torch.masked_fill(input1, mask, value)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1, mask, value):
        output = input1.masked_fill_(mask, value)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1, mask, value):
        if torch.is_tensor(value) and value.device.type != 'npu':
            value = value.to("npu")
        output = input1.masked_fill_(mask, value)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_masked_fill_shape_format_fp16(self, device="npu"):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [1.25,
                      torch.tensor(1.25, dtype=torch.float32),
                      torch.tensor(5, dtype=torch.int32),
                      torch.tensor(5, dtype=torch.int64)]

        shape_format = [[[np.float16, i, j], v] for i in format_list for j in shape_list for v in value_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            mask_cpu, mask_npu = self.create_bool_tensor(item[0][2], 0, 1)
            cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output1 = self.cpu_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output1 = self.npu_op_exec(npu_input1, mask_npu, item[1])
            cpu_output1 = cpu_output1.astype(npu_output1.dtype)
            self.assertRtolEqual(cpu_output1, npu_output1)

            cpu_output2 = self.cpu_inp_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output2 = self.npu_inp_op_exec(npu_input1, mask_npu, item[1])
            cpu_output2 = cpu_output2.astype(npu_output2.dtype)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_masked_fill_shape_format_fp32(self, device="npu"):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [1.25,
                      torch.tensor(1.25, dtype=torch.float32),
                      torch.tensor(5, dtype=torch.int32),
                      torch.tensor(5, dtype=torch.int64)]

        shape_format = [[[np.float32, i, j], v] for i in format_list for j in shape_list for v in value_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            mask_cpu, mask_npu = self.create_bool_tensor(item[0][2], 0, 1)

            cpu_output1 = self.cpu_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output1 = self.npu_op_exec(npu_input1, mask_npu, item[1])
            self.assertRtolEqual(cpu_output1, npu_output1)

            cpu_output2 = self.cpu_inp_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output2 = self.npu_inp_op_exec(npu_input1, mask_npu, item[1])
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_masked_fill_shape_format_int32(self, device="npu"):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [1.25,
                      torch.tensor(1.25, dtype=torch.float32),
                      torch.tensor(5, dtype=torch.int32),
                      torch.tensor(5, dtype=torch.int64)]

        shape_format = [[[np.int32, i, j], v] for i in format_list for j in shape_list for v in value_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            mask_cpu, mask_npu = self.create_bool_tensor(item[0][2], 0, 1)

            cpu_output1 = self.cpu_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output1 = self.npu_op_exec(npu_input1, mask_npu, item[1])
            self.assertRtolEqual(cpu_output1, npu_output1)

            cpu_output2 = self.cpu_inp_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output2 = self.npu_inp_op_exec(npu_input1, mask_npu, item[1])
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_masked_fill_shape_format_int64(self, device="npu"):
        format_list = [0]
        shape_list = [[1024], [32, 1024], [32, 8, 1024], [128, 32, 8, 1024]]
        value_list = [1.25,
                      torch.tensor(1.25, dtype=torch.float32),
                      torch.tensor(5, dtype=torch.int32),
                      torch.tensor(5, dtype=torch.int64)]

        shape_format = [[[np.int64, i, j], v] for i in format_list for j in shape_list for v in value_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            mask_cpu, mask_npu = self.create_bool_tensor(item[0][2], 0, 1)

            cpu_output1 = self.cpu_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output1 = self.npu_op_exec(npu_input1, mask_npu, item[1])
            cpu_output1 = cpu_output1.astype(np.int32)
            npu_output1 = npu_output1.astype(np.int32)
            self.assertRtolEqual(cpu_output1, npu_output1)

            cpu_output2 = self.cpu_inp_op_exec(cpu_input1, mask_cpu, item[1])
            npu_output2 = self.npu_inp_op_exec(npu_input1, mask_npu, item[1])
            cpu_output2 = cpu_output2.astype(np.int32)
            npu_output2 = npu_output2.astype(np.int32)
            self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
