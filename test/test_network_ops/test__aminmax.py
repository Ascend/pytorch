# Copyright (c) 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAminmax(TestCase):
    def cpu_op_exec(self, input0):
        output = torch._aminmax(input0)
        output0 = output[0].numpy()
        output1 = output[1].numpy()
        return output0, output1

    def npu_op_exec(self, input0):
        output = torch._aminmax(input0)
        output0 = output[0].cpu().numpy()
        output1 = output[1].cpu().numpy()
        return output0, output1

    def cpu_op_dim_exec(self, input0, dim, keepdim):
        output = torch._aminmax(input0, dim, keepdim)
        output0 = output[0].numpy()
        output1 = output[1].numpy()
        return output0, output1

    def npu_op_dim_exec(self, input0, dim, keepdim):
        output = torch._aminmax(input0, dim, keepdim)
        output0 = output[0].cpu().numpy()
        output1 = output[1].cpu().numpy()
        return output0, output1

    def cpu_op_exec_out(self, input0, in_dim, in_keepdim, min_in, max_in):
        output = torch.aminmax(input0, dim=in_dim, keepdim=in_keepdim, out=(min_in, max_in))
        output0 = output[0].numpy()
        output1 = output[1].numpy()
        return output0, output1

    def npu_op_exec_out(self, input0, in_dim, in_keepdim, min_in, max_in):
        output = torch.aminmax(input0, dim=in_dim, keepdim=in_keepdim, out=(min_in, max_in))
        output0 = output[0].cpu().numpy()
        output1 = output[1].cpu().numpy()
        return output0, output1

    def test__aminmax_shape_format(self):
        shape_format = [
            [np.float16, 0, [256, 1000]],
            [np.float32, 0, [1000]],
            [np.int8, 0, [256, 1000, 4, 4]],
            [np.int16, 0, [1000, 128, 3]],
            [np.int32, 0, [256]],
            [np.uint8, 0, [100, 128, 1000]],
            [np.int64, 0, [100, 128, 1000]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            if cpu_input.dtype == torch.half:
                cpu_input = cpu_input.to(torch.float)

            cpu_output0, cpu_output1 = self.cpu_op_exec(cpu_input)
            npu_output0, npu_output1 = self.npu_op_exec(npu_input)

            self.assertRtolEqual(cpu_output0.astype(npu_output0.dtype), npu_output0)
            self.assertRtolEqual(cpu_output1.astype(npu_output1.dtype), npu_output1)

    def test__aminmax_dim_shape_format(self):
        shape_format = [
            [np.float16, 0, [64, 4]],
            [np.float32, 0, [32, 4, 16, 8]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            dim = np.random.randint(len(item[2]))
            keepdim = np.random.randint(10) > 4
            if cpu_input.dtype == torch.half:
                cpu_input = cpu_input.to(torch.float)

            cpu_output0, cpu_output1 = self.cpu_op_dim_exec(cpu_input, dim, keepdim)
            npu_output0, npu_output1 = self.npu_op_dim_exec(npu_input, dim, keepdim)

            self.assertRtolEqual(cpu_output0.astype(npu_output0.dtype), npu_output0)
            self.assertRtolEqual(cpu_output1.astype(npu_output1.dtype), npu_output1)

    def test__aminmax_out_shape_format(self):
        shape_format = [
            [np.float16, 0, [64, 4]],
            [np.float32, 0, [32, 4, 16, 8]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            dim = np.random.randint(len(item[2]))
            keepdim = np.random.randint(10) > 4
            if cpu_input.dtype == torch.half:
                cpu_input = cpu_input.to(torch.float)

            out_temp = torch._aminmax(cpu_input, dim, keepdim)[0]
            out_size = out_temp.size()
            cpu_out_min = torch.zeros(out_size).to(cpu_input.dtype)
            cpu_out_max = torch.zeros(out_size).to(cpu_input.dtype)
            npu_out_min = torch.zeros(out_size).npu().to(npu_input.dtype)
            npu_out_max = torch.zeros(out_size).npu().to(npu_input.dtype)

            cpu_output0, cpu_output1 = self.cpu_op_exec_out(cpu_input, dim, keepdim, cpu_out_min, cpu_out_max)
            npu_output0, npu_output1 = self.npu_op_exec_out(npu_input, dim, keepdim, npu_out_min, npu_out_max)

            self.assertRtolEqual(cpu_output0.astype(npu_output0.dtype), npu_output0)
            self.assertRtolEqual(cpu_output1.astype(npu_output1.dtype), npu_output1)


if __name__ == "__main__":
    run_tests()
