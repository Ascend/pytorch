# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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


class TestQr(TestCase):
    def cpu_op_exec(self, input1, some):
        q, r = torch.qr(input1, some)
        return q.numpy(), r.numpy()

    def cpu_op_exec_tuple(self, input1, some):
        out = torch.qr(input1, some)
        output_q = out.Q
        output_r = out.R
        output_q = output_q.numpy()
        output_r = output_r.numpy()
        return output_q, output_r, out

    def npu_op_exec(self, input1, some):
        q, r = torch.qr(input1, some)
        qout = q.to("cpu").numpy()
        rout = r.to("cpu").numpy()
        return qout, rout

    def npu_op_exec_tuple(self, input1, some):
        out = torch.qr(input1.to("npu"), some)
        output_q = out.Q
        output_r = out.R
        output_q = output_q.to("cpu")
        output_r = output_r.to("cpu")
        output_q = output_q.numpy()
        output_r = output_r.numpy()
        return output_q, output_r, out

    def npu_op_exec_out(self, input1, some, input2, input3):
        torch.qr(input1, some, out=(input2, input3))
        qout = input2.to("cpu").numpy()
        rout = input3.to("cpu").numpy()
        return qout, rout

    def test_qr_shape_format(self, device="npu"):
        # TODO(ascend): 算子目前 暂不支持fp16, 后续开发中
        dtype_list = [np.float32]
        format_list = [-1]
        # Note:
        # precision may be lost if the magnitudes of the elements of input are large
        shape_list = [
            [3, 4],
            [2, 30, 30],
            [20, 10, 20],
            [8, 6, 50, 20],
            [10, 4, 6, 15, 13]
        ]
        somes_list = [True, False]
        shape_format = [
            [[d, i, j], l] for d in dtype_list for i in format_list
            for j in shape_list for l in somes_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            npu_input2 = torch.empty(0).npu().to(cpu_input1.dtype)
            npu_input3 = torch.empty(0).npu().to(cpu_input1.dtype)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1, cpu_output2 = self.cpu_op_exec(cpu_input1, item[1])
            npu_output1, npu_output2 = self.npu_op_exec(npu_input1, item[1])
            npu_output1_out, npu_output2_out = self.npu_op_exec_out(npu_input1, item[1], npu_input2, npu_input3)
            cpu_output1 = cpu_output1.astype(npu_output1.dtype)
            cpu_output2 = cpu_output2.astype(npu_output2.dtype)

            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(npu_output1_out, npu_output1)
            self.assertRtolEqual(npu_output2_out, npu_output2)

    def test_qr_common_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, -1, (5, 3)],
            [np.float32, -1, (1, 64, 147, 147)],
            [np.float32, -1, (65536, 14, 7, 1)],
            [np.int32, -1, (1000000, 3, 3, 1)],
            [np.int32, -1, (1024, 107, 31, 2)],
            [np.int32, -1, (1, 128, 1, 1)]
        ]
        somes = [True, False]
        for item in shape_format:
            for some in somes:
                cpu_input1, npu_input1 = create_common_tensor(item, -0.001, 0.001)
                if cpu_input1.dtype == torch.int32:
                    cpu_input1 = cpu_input1.to(torch.float32)
                if npu_input1.dtype == torch.int32:
                    npu_input1 = npu_input1.to(torch.float32)
                cpu_output_q, cpu_output_r, cpu_out = self.cpu_op_exec_tuple(cpu_input1, some)
                npu_output_q, npu_output_r, npu_out = self.npu_op_exec_tuple(npu_input1, some)
                npu_output = np.matmul(npu_output_q, npu_output_r)

                self.assertRtolEqual(cpu_output_q, npu_output_q)
                self.assertRtolEqual(cpu_output_r, npu_output_r)
                self.assertRtolEqual(cpu_input1.numpy(), npu_output)


if __name__ == "__main__":
    run_tests()
